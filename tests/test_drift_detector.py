"""
tests/test_drift_detector.py
────────────────────────────
Unit tests for DriftDetector (12 tests).

Key facts from the real implementation
---------------------------------------
- record(class_name: str, confidence: float) — confidence scores, NOT embeddings
- __init__ takes baseline_path pointing to a JSON file; .npy files sit alongside it
- JSON format: {"classes": {"cls": {"scores_file": "cls_scores.npy"}, ...}, ...}
- KS triggered when: buf_len >= min_samples AND counter % KS_RUN_EVERY_N == 0
- get_status() returns {"summary": {...}, "classes": {"cls": {...}, ...}}
- Per-class keys AFTER a test: ks_stat, p_value, is_alert, n_live, n_baseline, tested_at
- Per-class keys BEFORE any test: buffer_size, min_samples_required, tested=False

synthetic_baseline fixture (in conftest.py)
-------------------------------------------
Returns Path to a JSON file with 3 classes ("person", "bicycle", "car"),
each with 30 confidence scores in [0, 1].  KS_MIN_LIVE_SAMPLES = 100.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import config as cfg
from src.monitoring.drift_detector import DriftDetector


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_detector(
    baseline_json: Path, min_samples: int = cfg.KS_MIN_LIVE_SAMPLES
) -> DriftDetector:
    """Instantiate DriftDetector with a no-op Redis mock."""
    redis_mock = MagicMock()
    redis_mock.available = False
    redis_mock.get_list.return_value = []
    return DriftDetector(
        baseline_path=baseline_json,
        redis_client=redis_mock,
        min_samples=min_samples,
        alert_threshold=cfg.KS_DRIFT_ALERT_THRESHOLD,
    )


def _fill_buffer(
    detector: DriftDetector, class_name: str, n: int, value: float = 0.65
) -> None:
    """Record n identical confidence scores for class_name."""
    for _ in range(n):
        detector.record(class_name, value)


# ── Instantiation ──────────────────────────────────────────────────────────────


def test_detector_loads_correct_number_of_classes(synthetic_baseline: Path) -> None:
    detector = _make_detector(synthetic_baseline)
    assert len(detector._baselines) == 3


def test_detector_loads_correct_sample_count(synthetic_baseline: Path) -> None:
    detector = _make_detector(synthetic_baseline)
    for cls_name, scores in detector._baselines.items():
        assert scores.shape == (30,), (
            f"{cls_name}: expected 30 scores, got {scores.shape}"
        )


def test_detector_missing_baseline_raises_file_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "nonexistent_baseline.json"
    redis_mock = MagicMock()
    redis_mock.available = False
    redis_mock.get_list.return_value = []
    with pytest.raises(FileNotFoundError):
        DriftDetector(
            baseline_path=missing_path,
            redis_client=redis_mock,
        )


# ── Buffer management ──────────────────────────────────────────────────────────


def test_record_appends_to_buffer(synthetic_baseline: Path) -> None:
    detector = _make_detector(synthetic_baseline)
    detector.record("person", 0.85)
    assert len(detector._buffers["person"]) == 1
    assert abs(list(detector._buffers["person"])[0] - 0.85) < 1e-6


def test_buffer_does_not_exceed_maxlen(synthetic_baseline: Path) -> None:
    detector = _make_detector(synthetic_baseline)
    for i in range(DriftDetector.MAX_BUFFER + 50):
        detector.record("person", float(i % 10) / 10)
    assert len(detector._buffers["person"]) == DriftDetector.MAX_BUFFER


def test_record_unknown_class_is_silently_ignored(synthetic_baseline: Path) -> None:
    detector = _make_detector(synthetic_baseline)
    detector.record("nonexistent_class", 0.9)  # must not raise
    assert "nonexistent_class" not in detector._buffers


# ── KS test triggering logic ───────────────────────────────────────────────────


def test_ks_not_triggered_below_min_samples(synthetic_baseline: Path) -> None:
    """KS test does not run when buffer has fewer than min_samples entries."""
    detector = _make_detector(synthetic_baseline, min_samples=100)
    _fill_buffer(detector, "person", n=99)  # one below threshold
    status = detector.get_status()
    assert status["classes"]["person"]["tested"] is False


def test_ks_triggered_at_min_samples_boundary(synthetic_baseline: Path) -> None:
    """KS test runs once min_samples reached AND counter is at a KS_RUN_EVERY_N multiple."""
    detector = _make_detector(synthetic_baseline, min_samples=cfg.KS_MIN_LIVE_SAMPLES)
    # Record exactly min_samples; counter=min_samples which is a multiple of KS_RUN_EVERY_N
    _fill_buffer(detector, "person", n=cfg.KS_MIN_LIVE_SAMPLES, value=0.65)
    status = detector.get_status()
    assert status["classes"]["person"]["tested"] is True


# ── Double-gate alert logic ────────────────────────────────────────────────────


def test_no_alert_on_same_distribution(synthetic_baseline: Path) -> None:
    """Same distribution → high p-value → no alert (double-gate prevents FP)."""
    detector = _make_detector(synthetic_baseline, min_samples=cfg.KS_MIN_LIVE_SAMPLES)
    # Load baseline scores and replay them as live scores
    baseline_scores = detector._baselines["person"]
    # Repeat baseline scores to fill min_samples (baseline has only 30)
    n_repeats = (cfg.KS_MIN_LIVE_SAMPLES // len(baseline_scores)) + 1
    for score in list(baseline_scores) * n_repeats:
        detector.record("person", float(score))
    status = detector.get_status()
    assert status["classes"]["person"]["is_alert"] is False


def test_alert_fires_on_heavily_shifted_distribution(synthetic_baseline: Path) -> None:
    """Strongly shifted distribution → low p-value → alert fires."""
    detector = _make_detector(synthetic_baseline, min_samples=cfg.KS_MIN_LIVE_SAMPLES)
    # Baseline "person" is ~N(0.6, 0.1); shift live to 0.99 (max confidence)
    _fill_buffer(detector, "person", n=cfg.KS_MIN_LIVE_SAMPLES, value=0.99)
    status = detector.get_status()
    cls_status = status["classes"]["person"]
    assert cls_status["is_alert"] is True
    assert cls_status["p_value"] < 0.05


# ── get_status() schema ────────────────────────────────────────────────────────


def test_get_status_before_records_has_tested_false(synthetic_baseline: Path) -> None:
    detector = _make_detector(synthetic_baseline)
    status = detector.get_status()
    for cls_name in ["person", "bicycle", "car"]:
        cls_status = status["classes"][cls_name]
        assert cls_status["tested"] is False
        assert cls_status["buffer_size"] == 0


def test_get_status_after_ks_has_required_keys(synthetic_baseline: Path) -> None:
    """After a KS test runs, per-class dict has all expected keys."""
    detector = _make_detector(synthetic_baseline, min_samples=cfg.KS_MIN_LIVE_SAMPLES)
    _fill_buffer(detector, "bicycle", n=cfg.KS_MIN_LIVE_SAMPLES)
    status = detector.get_status()
    cls_status = status["classes"]["bicycle"]
    assert cls_status["tested"] is True
    for key in ("ks_stat", "p_value", "is_alert", "n_live", "n_baseline"):
        assert key in cls_status, f"Missing key: {key}"


def test_p_value_in_zero_to_one_range(synthetic_baseline: Path) -> None:
    detector = _make_detector(synthetic_baseline, min_samples=cfg.KS_MIN_LIVE_SAMPLES)
    _fill_buffer(detector, "car", n=cfg.KS_MIN_LIVE_SAMPLES)
    status = detector.get_status()
    p = status["classes"]["car"]["p_value"]
    assert 0.0 <= p <= 1.0


def test_is_alert_is_bool_not_int(synthetic_baseline: Path) -> None:
    """is_alert must be a Python bool (True/False), not 0/1 integer."""
    detector = _make_detector(synthetic_baseline, min_samples=cfg.KS_MIN_LIVE_SAMPLES)
    _fill_buffer(detector, "person", n=cfg.KS_MIN_LIVE_SAMPLES)
    status = detector.get_status()
    alert_val = status["classes"]["person"]["is_alert"]
    assert isinstance(alert_val, bool), f"Expected bool, got {type(alert_val)}"
