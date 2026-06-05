# filename: streamlit_app/api_client.py
# purpose:  Thin HTTP client wrapping FastAPI endpoints. Module-level singleton session —
#           no @st.cache_resource dependency so this is safely importable from notebooks/tests.
#           Every public function converts all exceptions to RuntimeError before returning,
#           so pages only need to catch RuntimeError and call st.error(str(e)).

from __future__ import annotations

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

import sys
from pathlib import Path

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Base URL ──────────────────────────────────────────────────────────────────
FASTAPI_URL: str = config.FASTAPI_URL or "http://localhost:8000"

# ── Per-endpoint timeouts (seconds) ───────────────────────────────────────────
HEALTH_TIMEOUT   = 3    # liveness check — fail fast
CLASSIFY_TIMEOUT = 30   # cold CPU inference can be slow
ENSEMBLE_TIMEOUT = 90   # 3 models in sequence on CPU
DETECT_TIMEOUT   = 45   # YOLO forward pass + NMS on CPU
DRIFT_TIMEOUT    = 5    # status read should always be fast

# ── Module-level singleton session ────────────────────────────────────────────
_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers["User-Agent"] = "SmartVision-Streamlit/1.0"
    return _SESSION


# ── Centralised error handler ─────────────────────────────────────────────────
def _handle_error(e: Exception, endpoint: str) -> None:
    if isinstance(e, ConnectionError):
        raise RuntimeError(
            f"FastAPI not reachable at {FASTAPI_URL}. "
            "Start uvicorn with: uvicorn api.main:app --reload"
        )
    if isinstance(e, Timeout):
        raise RuntimeError(
            f"FastAPI request timed out on {endpoint}. "
            "Check that models are loaded (/health)."
        )
    if isinstance(e, HTTPError):
        raise RuntimeError(
            f"FastAPI returned {e.response.status_code} on {endpoint}: "
            f"{e.response.text[:300]}"
        )
    # Catch-all: JSONDecodeError, ChunkedEncodingError, TooManyRedirects, SSLError, etc.
    raise RuntimeError(
        f"Unexpected error calling {endpoint}: {type(e).__name__}: {e}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_health() -> dict:
    """GET /health — returns {"status": ..., "models_ready": bool, "models_loaded": [...]}."""
    try:
        resp = _session().get(f"{FASTAPI_URL}/health", timeout=HEALTH_TIMEOUT)
        # Do NOT raise_for_status here: 503 during loading is valid and has a useful body.
        return resp.json()
    except Exception as e:
        _handle_error(e, "/health")


def classify(img_bytes: bytes, filename: str, model_name: str, top_k: int) -> dict:
    """POST /classify — returns ClassifyResponse dict."""
    try:
        resp = _session().post(
            f"{FASTAPI_URL}/classify",
            files={"file": (filename, img_bytes, "image/jpeg")},
            data={"model_name": model_name, "top_k": str(top_k)},
            timeout=CLASSIFY_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        _handle_error(e, "/classify")


def ensemble(img_bytes: bytes, filename: str, top_k: int) -> dict:
    """POST /ensemble — weighted average of ResNet50 + EfficientNetB0 + MobileNetV2."""
    try:
        resp = _session().post(
            f"{FASTAPI_URL}/ensemble",
            files={"file": (filename, img_bytes, "image/jpeg")},
            data={"top_k": str(top_k)},
            timeout=ENSEMBLE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        _handle_error(e, "/ensemble")


def detect(img_bytes: bytes, filename: str, conf_threshold: float) -> dict:
    """POST /detect — returns DetectResponse dict with absolute-pixel bbox coords."""
    try:
        resp = _session().post(
            f"{FASTAPI_URL}/detect",
            files={"file": (filename, img_bytes, "image/jpeg")},
            data={"conf_threshold": str(conf_threshold)},
            timeout=DETECT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        _handle_error(e, "/detect")


def get_drift_status() -> dict:
    """GET /drift/status — returns DriftStatusResponse dict."""
    try:
        resp = _session().get(f"{FASTAPI_URL}/drift/status", timeout=DRIFT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        _handle_error(e, "/drift/status")
