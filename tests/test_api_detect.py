"""
tests/test_api_detect.py
────────────────────────
Tests for POST /detect endpoint (10 tests).

YOLO mock architecture
-----------------------
detect.py calls:
    results = yolo.predict(img_array, conf=conf_threshold, verbose=False)
    for box in results[0].boxes:
        box.cls[0].item()    → class index (int)
        box.conf[0].item()   → confidence (float)
        box.xyxy[0][i].item() → bbox coordinates (float)

MockYOLOBox provides real torch.Tensor values so .item() works correctly.
The mock YOLO model returns a list with one Results-like object.

Conf-threshold filtering is performed by YOLO internally; the endpoint
passes conf_threshold directly to yolo.predict and does NOT re-filter.
Tests verify the conf_threshold is forwarded correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import pytest
import torch

from api.schemas import DetectResponse

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────────────

class MockYOLOBox:
    """Minimal YOLO box with real tensors so .item() works."""

    def __init__(
        self,
        class_id: int = 0,
        conf: float = 0.92,
        x1: float = 10.0,
        y1: float = 20.0,
        x2: float = 50.0,
        y2: float = 80.0,
    ):
        self.cls = [torch.tensor(float(class_id))]
        self.conf = [torch.tensor(conf)]
        self.xyxy = [torch.tensor([x1, y1, x2, y2])]


def _make_yolo_mock(boxes: list[MockYOLOBox] | None = None) -> MagicMock:
    """Build a YOLO model mock returning the given boxes."""
    if boxes is None:
        boxes = []
    result = MagicMock()
    result.names = {0: "person", 1: "bicycle", 2: "car"}
    result.boxes = boxes
    yolo = MagicMock()
    yolo.predict.return_value = [result]
    return yolo


def _post_detect(
    client: "TestClient",
    image_bytes: bytes,
    conf_threshold: float = 0.5,
) -> "TestClient":
    return client.post(
        "/detect",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        data={"conf_threshold": conf_threshold},
    )


# ── Cache hit / miss ───────────────────────────────────────────────────────────

def test_detect_cache_miss_returns_cached_false(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """Cache miss: YOLO inference runs, cached=False."""
    app_client.app.state.models["yolo"] = _make_yolo_mock()
    app_client.app.state.redis.get.return_value = None
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 200
        assert response.json()["cached"] is False
    finally:
        app_client.app.state.models.pop("yolo", None)


def test_detect_cache_hit_returns_cached_true(
    app_client: "TestClient",
    test_image_bytes: bytes,
    mock_detect_response: dict,
) -> None:
    """Cache hit: stored payload returned with cached=True, YOLO not called."""
    app_client.app.state.models["yolo"] = _make_yolo_mock()
    app_client.app.state.redis.get.return_value = mock_detect_response
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 200
        assert response.json()["cached"] is True
    finally:
        app_client.app.state.models.pop("yolo", None)


# ── Confidence threshold forwarded to YOLO ────────────────────────────────────

def test_detect_conf_threshold_forwarded_to_yolo(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """Endpoint passes conf_threshold to yolo.predict (YOLO handles filtering)."""
    mock_yolo = _make_yolo_mock()
    app_client.app.state.models["yolo"] = mock_yolo
    app_client.app.state.redis.get.return_value = None
    try:
        _post_detect(app_client, test_image_bytes, conf_threshold=0.9)
        call_kwargs = mock_yolo.predict.call_args[1]
        assert call_kwargs["conf"] == pytest.approx(0.9)
    finally:
        app_client.app.state.models.pop("yolo", None)


def test_detect_empty_detections_when_yolo_returns_none(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """YOLO returns zero boxes → detections list is empty."""
    app_client.app.state.models["yolo"] = _make_yolo_mock(boxes=[])
    app_client.app.state.redis.get.return_value = None
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 200
        assert response.json()["detections"] == []
    finally:
        app_client.app.state.models.pop("yolo", None)


# ── 503 error conditions ───────────────────────────────────────────────────────

def test_detect_503_when_models_not_ready(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    app_client.app.state.models_ready = False
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 503
    finally:
        app_client.app.state.models_ready = True


def test_detect_503_when_yolo_model_missing(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """Default app_client has resnet50 + mobilenet but NO yolo → 503."""
    assert "yolo" not in app_client.app.state.models
    response = _post_detect(app_client, test_image_bytes)
    assert response.status_code == 503
    assert "detection model" in response.json()["detail"].lower()


# ── Bbox coordinate correctness ────────────────────────────────────────────────

def test_detect_bbox_x1_lt_x2_and_y1_lt_y2(
    app_client: "TestClient",
    test_image_bytes: bytes,
    mock_detect_response: dict,
) -> None:
    """All returned bboxes satisfy x1 < x2 and y1 < y2."""
    app_client.app.state.models["yolo"] = _make_yolo_mock()
    app_client.app.state.redis.get.return_value = mock_detect_response
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 200
        for det in response.json()["detections"]:
            assert det["x1"] < det["x2"], f"x1 >= x2: {det}"
            assert det["y1"] < det["y2"], f"y1 >= y2: {det}"
    finally:
        app_client.app.state.models.pop("yolo", None)


# ── Response body correctness ──────────────────────────────────────────────────

def test_detect_inference_time_is_positive(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    app_client.app.state.models["yolo"] = _make_yolo_mock()
    app_client.app.state.redis.get.return_value = None
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 200
        assert response.json()["inference_time_ms"] > 0
    finally:
        app_client.app.state.models.pop("yolo", None)


def test_detect_response_validates_schema(
    app_client: "TestClient",
    test_image_bytes: bytes,
    mock_detect_response: dict,
) -> None:
    """Response dict validates against DetectResponse Pydantic schema."""
    app_client.app.state.models["yolo"] = _make_yolo_mock()
    app_client.app.state.redis.get.return_value = mock_detect_response
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 200
        parsed = DetectResponse(**response.json())
        assert parsed.cached is True
    finally:
        app_client.app.state.models.pop("yolo", None)


# ── Input validation ───────────────────────────────────────────────────────────

def test_detect_non_image_bytes_rejected(app_client: "TestClient") -> None:
    """Garbage bytes that PIL cannot open → error before YOLO is called."""
    app_client.app.state.models["yolo"] = _make_yolo_mock()
    try:
        response = app_client.post(
            "/detect",
            files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
            data={"conf_threshold": 0.5},
        )
        assert response.status_code in (400, 422, 500)
    finally:
        app_client.app.state.models.pop("yolo", None)


def test_detect_real_box_coordinates_extracted_correctly(
    app_client: "TestClient", test_image_bytes: bytes
) -> None:
    """When YOLO returns a real box, coordinates appear correctly in response."""
    box = MockYOLOBox(class_id=0, conf=0.88, x1=5.0, y1=10.0, x2=100.0, y2=200.0)
    mock_yolo = _make_yolo_mock(boxes=[box])
    app_client.app.state.models["yolo"] = mock_yolo
    app_client.app.state.redis.get.return_value = None
    try:
        response = _post_detect(app_client, test_image_bytes)
        assert response.status_code == 200
        dets = response.json()["detections"]
        assert len(dets) == 1
        d = dets[0]
        assert d["class_name"] == "person"   # names[0] = "person"
        assert d["confidence"] == pytest.approx(0.88, abs=1e-4)
        assert d["x1"] == pytest.approx(5.0)
        assert d["y1"] == pytest.approx(10.0)
        assert d["x2"] == pytest.approx(100.0)
        assert d["y2"] == pytest.approx(200.0)
    finally:
        app_client.app.state.models.pop("yolo", None)
