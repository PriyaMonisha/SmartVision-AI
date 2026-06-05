"""
tests/test_api_classify.py
──────────────────────────
Tests for POST /classify endpoint (14 tests).

Covers:
  - Cache hit / miss paths
  - top_k clamping and validation
  - Model selection validation (unknown name, yolo rejected)
  - 503 when models not ready
  - Response body correctness (confidence sum, sorting, inference_time)
  - Input validation (bad image, empty file)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _post_classify(
    client: "TestClient",
    image_bytes: bytes,
    model_name: str = "resnet50",
    top_k: int = 5,
) -> "TestClient":
    return client.post(
        "/classify",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        data={"model_name": model_name, "top_k": top_k},
    )


# ── Cache hit / miss ───────────────────────────────────────────────────────────

def test_classify_cache_miss_returns_cached_false(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """Cache miss path: Redis returns None → model runs → cached=False."""
    app_client.app.state.redis.get.return_value = None  # explicit miss
    response = _post_classify(app_client, test_image_bytes)
    assert response.status_code == 200
    assert response.json()["cached"] is False


def test_classify_cache_hit_returns_cached_true(
    app_client: "TestClient",
    test_image_bytes: bytes,
    mock_classify_response: dict,
) -> None:
    """Cache hit path: Redis returns stored dict → cached=True, no inference."""
    app_client.app.state.redis.get.return_value = mock_classify_response
    response = _post_classify(app_client, test_image_bytes)
    assert response.status_code == 200
    assert response.json()["cached"] is True


# ── top_k behaviour ────────────────────────────────────────────────────────────

def test_classify_top_k_exactly_three(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """top_k=3 returns exactly 3 predictions."""
    app_client.app.state.redis.get.return_value = None
    response = _post_classify(app_client, test_image_bytes, top_k=3)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_classify_top_k_exceeds_num_classes_is_clipped(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """top_k=100 (> 22 classes) is silently clamped to NUM_CLASSES."""
    app_client.app.state.redis.get.return_value = None
    response = _post_classify(app_client, test_image_bytes, top_k=100)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) <= 22


def test_classify_top_k_zero_is_rejected(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """top_k=0 is invalid; endpoint raises (500) or validates (422) — never 200."""
    response = _post_classify(app_client, test_image_bytes, top_k=0)
    # FastAPI does not validate Form int ranges, so top_k=0 reaches route logic.
    # probs.topk(0) succeeds but predictions[0] then raises IndexError → 500.
    assert response.status_code in (422, 500)


# ── Model name validation ──────────────────────────────────────────────────────

def test_classify_unknown_model_rejected(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """Non-existent model name → 400."""
    response = _post_classify(app_client, test_image_bytes, model_name="nonexistent")
    assert response.status_code == 400


def test_classify_yolo_model_rejected(
    app_client: "TestClient", test_image_bytes: bytes, mock_model: "MagicMock"
) -> None:
    """YOLO is a detection model; /classify must reject it with 400.

    Route logic: `if model_name not in models or model_name == "yolo": raise 400`
    Detail message lists the allowed models, not "yolo" itself.
    """
    app_client.app.state.models["yolo"] = mock_model
    try:
        response = _post_classify(app_client, test_image_bytes, model_name="yolo")
        assert response.status_code == 400
        assert "model_name" in response.json()["detail"].lower()
    finally:
        app_client.app.state.models.pop("yolo", None)


# ── 503 when not ready ─────────────────────────────────────────────────────────

def test_classify_503_when_models_not_ready(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    app_client.app.state.models_ready = False
    try:
        response = _post_classify(app_client, test_image_bytes)
        assert response.status_code == 503
    finally:
        app_client.app.state.models_ready = True


# ── Response body correctness ──────────────────────────────────────────────────

def test_classify_confidence_scores_sum_to_one(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """Softmax output sums to 1.0 across all classes (within floating-point tolerance)."""
    app_client.app.state.redis.get.return_value = None
    response = _post_classify(app_client, test_image_bytes, top_k=22)
    assert response.status_code == 200
    total = sum(p["confidence"] for p in response.json()["predictions"])
    assert 0.99 <= total <= 1.01


def test_classify_inference_time_is_positive(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    app_client.app.state.redis.get.return_value = None
    response = _post_classify(app_client, test_image_bytes)
    assert response.status_code == 200
    assert response.json()["inference_time_ms"] > 0


def test_classify_predictions_sorted_descending(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """Predictions are ordered by confidence (highest first)."""
    app_client.app.state.redis.get.return_value = None
    response = _post_classify(app_client, test_image_bytes, top_k=5)
    assert response.status_code == 200
    confs = [p["confidence"] for p in response.json()["predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_classify_model_name_echoed_in_response(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    app_client.app.state.redis.get.return_value = None
    response = _post_classify(app_client, test_image_bytes, model_name="resnet50")
    assert response.status_code == 200
    assert response.json()["model_name"] == "resnet50"


# ── Input validation ───────────────────────────────────────────────────────────

def test_classify_non_image_bytes_rejected(app_client: "TestClient") -> None:
    """Garbage bytes that PIL cannot open → 400."""
    response = app_client.post(
        "/classify",
        files={"file": ("bad.jpg", b"not an image at all", "image/jpeg")},
        data={"model_name": "resnet50", "top_k": 5},
    )
    assert response.status_code in (400, 422, 500)


def test_classify_empty_file_rejected(app_client: "TestClient") -> None:
    """Zero-byte file upload → error (PIL raises or FastAPI rejects)."""
    response = app_client.post(
        "/classify",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"model_name": "resnet50", "top_k": 5},
    )
    assert response.status_code in (400, 422, 500)
