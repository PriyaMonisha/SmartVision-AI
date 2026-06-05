# filename: src/monitoring/drift_detector.py
# purpose:  KS drift detection on live classifier confidence scores.
#           Compares rolling live buffer against val-split baseline (Section 7).
#
# Rules applied:
#   Rule 4:  src/ never imports from api/ — RedisCache injected as constructor arg
#   Rule 21: KS test on confidence scores (not raw embeddings)
#   Rule 23: Redis optional — deque fallback when unavailable
#   Rule 33: label name "class_name" consistent across all Prometheus metrics

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from prometheus_client import Gauge
from scipy.stats import ks_2samp

import config as cfg

if TYPE_CHECKING:
    from src.inference.redis_cache import RedisCache

logger = logging.getLogger(__name__)

# ── Prometheus Gauges — module-level singletons ───────────────────────────────
# Defined here (not api/prometheus_metrics.py) because Rule 4 prohibits src/
# from importing api/. The global prometheus_client registry collects all metrics
# regardless of definition location — GET /metrics still exposes all of them.
ks_drift_statistic = Gauge(
    "smartvision_ks_drift_statistic",
    "KS D-statistic from last test (0=no drift, 1=complete separation)",
    ["class_name"],
)
ks_drift_p_value = Gauge(
    "smartvision_ks_drift_p_value",
    "KS p-value from last test (1.0=no evidence of drift, <0.05=drift likely)",
    ["class_name"],
)
ks_drift_alert = Gauge(
    "smartvision_ks_drift_alert",
    "1 if KS stat > threshold AND p < 0.05, else 0 (double-gate prevents false positives)",
    ["class_name"],
)
live_buffer_size = Gauge(
    "smartvision_live_buffer_size",
    "Number of live inference scores in rolling buffer",
    ["class_name"],
)

# ── Rate-limit constant ───────────────────────────────────────────────────────
KS_RUN_EVERY_N: int = cfg.KS_RUN_EVERY_N  # run KS every N new samples per class


# ── Helper ────────────────────────────────────────────────────────────────────


def _safe_float(v: object) -> Optional[float]:
    """Return Python float or None for nan/inf (prevents json.dumps failure).

    scipy.stats.ks_2samp returns nan pvalue on zero-variance live distributions
    on some scipy versions. Storing None propagates cleanly to Pydantic Optional[float].
    """
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if (f == f and abs(f) != float("inf")) else None
    except (TypeError, ValueError):
        return None


# ── DriftDetector ─────────────────────────────────────────────────────────────


class DriftDetector:
    """Per-class KS drift detector for classifier confidence scores.

    Architecture
    ------------
    - Baseline: val-split MobileNet confidence scores loaded from .npy files
      (22 classes × 30 samples, generated in Section 7).
    - Live buffer: rolling deque(maxlen=200) per class, optionally persisted to
      Redis list ``sv:drift:live:{class_name}`` so state survives API restarts.
    - KS test: runs every KS_RUN_EVERY_N new samples once buffer >= min_samples.
      Rate-limit prevents blocking the asyncio event loop on high-traffic paths.
    - Alert: double-gate (stat > threshold AND p < 0.05) to avoid ~95% false-
      positive rate with n=30 baseline samples.

    Thread safety
    -------------
    ``record()`` and ``get_status()`` are both synchronous and run in the asyncio
    event loop thread. Dict assignments in ``_run_ks()`` are GIL-protected single
    operations. DO NOT add ``await`` points to ``record()`` or ``_run_ks()``, and
    DO NOT call them via ``run_in_executor()``, without adding a ``threading.Lock``
    around ``_last_results`` writes.
    """

    MAX_BUFFER = 200  # maximum live scores kept per class

    def __init__(
        self,
        baseline_path: Path,
        redis_client: "RedisCache",
        min_samples: int = 100,
        alert_threshold: float = 0.10,
    ) -> None:
        # Baseline JSON is required — raise immediately if missing (not graceful-degrade)
        baseline_path = Path(baseline_path)
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Drift baseline not found: {baseline_path}. "
                "Run notebooks/06_model_comparison.py to generate it."
            )

        self._min_samples = min_samples
        self._alert_threshold = alert_threshold
        self._redis = redis_client

        # Load baseline metadata
        with baseline_path.open() as f:
            self._baseline_meta: dict = json.load(f)

        # Resolve .npy paths relative to the JSON file's directory (not CWD)
        baseline_dir = baseline_path.parent
        self._baselines: dict[str, np.ndarray] = {}
        for cls_name, entry in self._baseline_meta.get("classes", {}).items():
            npy_path = baseline_dir / entry["scores_file"]
            if not npy_path.exists():
                logger.warning(
                    f"Baseline scores missing for '{cls_name}': {npy_path}. Skipping."
                )
                continue
            self._baselines[cls_name] = np.load(str(npy_path))

        logger.info(
            f"DriftDetector: loaded baseline for {len(self._baselines)} classes"
        )

        # Rolling buffers — deque provides O(1) append and automatic maxlen eviction
        self._buffers: dict[str, deque[float]] = {
            cls: deque(maxlen=self.MAX_BUFFER) for cls in self._baselines
        }
        # Per-class call counter for KS rate-limiting
        self._counters: dict[str, int] = {cls: 0 for cls in self._baselines}
        # Last KS result per class — written in _run_ks, read in get_status
        self._last_results: dict[str, dict] = {}

        # Step 9: Initialize all Gauge label series before any classify request
        # so Prometheus has time series from startup (alert rules evaluate empty sets otherwise)
        for cls in cfg.CLASSES:
            ks_drift_statistic.labels(class_name=cls).set(0.0)
            ks_drift_p_value.labels(class_name=cls).set(
                1.0
            )  # 1.0 = "no evidence of drift"
            ks_drift_alert.labels(class_name=cls).set(0.0)
            live_buffer_size.labels(class_name=cls).set(0.0)

        # Step 10-11: Restore buffers from Redis; update Gauges to reflect restored state
        for cls in self._baselines:
            redis_key = f"sv:drift:live:{cls}"
            restored = redis_client.get_list(redis_key)  # newest-first (LPUSH order)
            if restored:
                # Reverse to chronological order before extending deque so that
                # maxlen eviction removes the oldest item (not the most recently restored)
                self._buffers[cls].extend(reversed(restored))
                live_buffer_size.labels(class_name=cls).set(len(self._buffers[cls]))
                logger.info(
                    f"  Restored {len(self._buffers[cls])} scores for '{cls}' from Redis"
                )

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, class_name: str, confidence: float) -> None:
        """Record one live inference score; run KS if buffer threshold met.

        Called from classify route after every non-cached inference.
        Cache hits MUST NOT call record() — cached responses double-count scores.
        """
        if class_name not in self._baselines:
            return  # silently skip "unknown" and any class without baseline

        confidence = max(0.0, min(1.0, float(confidence)))  # clamp to [0, 1]

        self._buffers[class_name].append(confidence)

        # Push to Redis for cross-restart persistence (best-effort)
        if self._redis.available:
            self._redis.push_to_list(
                f"sv:drift:live:{class_name}", confidence, self.MAX_BUFFER
            )

        buf_len = len(self._buffers[class_name])
        live_buffer_size.labels(class_name=class_name).set(buf_len)

        self._counters[class_name] += 1
        if (
            buf_len >= self._min_samples
            and self._counters[class_name] % KS_RUN_EVERY_N == 0
        ):
            self._run_ks(class_name, list(self._buffers[class_name]))

    def get_status(self) -> dict:
        """Return per-class KS results + summary counts for /drift/status.

        All numeric values are Python native types (float/int/bool).
        ``_safe_float`` in ``_run_ks`` ensures no np.float32 or nan/inf leaks here.
        """
        classes_out: dict[str, dict] = {}
        for cls in self._baselines:
            buf_len = len(self._buffers[cls])
            result = self._last_results.get(cls)
            classes_out[cls] = {
                "buffer_size": buf_len,
                "min_samples_required": self._min_samples,
                "tested": result is not None,
                **(result if result is not None else {}),
            }

        alerting = [cls for cls, r in self._last_results.items() if r.get("is_alert")]
        return {
            "summary": {
                "total_classes": len(self._baselines),
                "classes_with_data": sum(
                    1 for b in self._buffers.values() if len(b) > 0
                ),
                "classes_tested": len(self._last_results),
                "classes_alerting": len(alerting),
                "baseline_model": self._baseline_meta.get("model_used", "mobilenet"),
                "baseline_split": self._baseline_meta.get("split_used", "val"),
            },
            "classes": classes_out,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_ks(self, class_name: str, live_scores: list[float]) -> None:
        """Run scipy KS test and update Prometheus Gauges + _last_results.

        THREADING NOTE: This method is intentionally synchronous and runs in the
        asyncio event loop thread (called only from record()). DO NOT add await
        points here or call from run_in_executor() without adding a threading.Lock
        around _last_results to prevent partial-update races.
        """
        baseline = self._baselines[class_name]
        live = np.array(live_scores, dtype=np.float32)

        result = ks_2samp(baseline, live)
        stat = float(result.statistic)
        pval = float(result.pvalue)

        # Double-gate alert: stat alone has ~95% false-positive rate with n=30 baseline
        # (KS variance is high at small reference sample sizes).
        # Requiring p < 0.05 as second gate eliminates false positives.
        is_alert = bool(stat > self._alert_threshold and pval < 0.05)

        # _safe_float converts nan/inf (degenerate inputs) to None so _last_results
        # is always JSON-safe without a round-trip in get_status()
        safe_stat = _safe_float(stat)
        safe_pval = _safe_float(pval)

        self._last_results[class_name] = {
            "ks_stat": safe_stat,
            "p_value": safe_pval,
            "is_alert": is_alert,
            "n_live": int(len(live_scores)),
            "n_baseline": int(len(baseline)),
            "tested_at": float(time.time()),
        }

        # Fall back to neutral Gauge values if stat/pval are nan/inf
        ks_drift_statistic.labels(class_name=class_name).set(
            safe_stat if safe_stat is not None else 0.0
        )
        ks_drift_p_value.labels(class_name=class_name).set(
            safe_pval if safe_pval is not None else 1.0
        )
        ks_drift_alert.labels(class_name=class_name).set(1.0 if is_alert else 0.0)

        if is_alert:
            logger.warning(
                f"Drift alert: class={class_name!r} stat={stat:.4f} p={pval:.6f} "
                f"n_live={len(live_scores)} n_baseline={len(baseline)}"
            )
