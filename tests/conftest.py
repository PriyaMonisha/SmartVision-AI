"""
tests/conftest.py
─────────────────
Shared pytest fixtures for SmartVision AI test suite.

Fixtures
--------
test_image_bytes     100×100 red JPEG as bytes (session-scoped, created once)
MockClassificationModel  torch.nn.Module returning fixed 22-class logits
mock_model           MockClassificationModel instance in eval() mode
mock_redis           MagicMock RedisCache (cache miss by default)
mock_classify_response  Sample ClassifyResponse-compatible dict (no "cached" field)
mock_detect_response    Sample DetectResponse-compatible dict (no "cached" field)
app_client           TestClient with fully mocked app.state
synthetic_baseline   JSON + .npy confidence-score baseline for DriftDetector tests
fake_redis_client    fakeredis.FakeStrictRedis for RedisCache unit tests
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


# ── Mock classification model ──────────────────────────────────────────────────

class MockClassificationModel(torch.nn.Module):
    """Real torch.nn.Module returning deterministic 22-class logits.

    Using a real Module (not MagicMock) ensures .eval(), .to(device), and
    forward() behave exactly as the route expects.
    """

    def __init__(self, num_classes: int = 22):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        logits = torch.zeros(batch, self.num_classes)
        logits[:, 0] = 5.0   # class 0 is always top-1
        logits[:, 1] = 2.0
        logits[:, 2] = 1.0
        return logits


# ── Session-scoped image bytes (created once for the entire test run) ──────────

@pytest.fixture(scope="session")
def test_image_bytes() -> bytes:
    """100×100 red JPEG as bytes. Session-scoped to avoid re-encoding per test."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.read()


# ── Model fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model() -> MockClassificationModel:
    return MockClassificationModel(num_classes=22).eval()


# ── Redis mock (for API-layer tests) ──────────────────────────────────────────

@pytest.fixture
def mock_redis() -> MagicMock:
    """MagicMock replacing RedisCache. Default: cache miss (get returns None)."""
    m = MagicMock()
    m.get.return_value = None
    m.set.return_value = None
    m.available = True
    m._available = True
    return m


# ── Pre-built response dicts (stored format — no "cached" field) ───────────────

@pytest.fixture
def mock_classify_response() -> dict:
    """Stored cache payload for /classify (cached=False already popped on store)."""
    from config import CLASSES
    return {
        "predictions": [
            {"class_name": CLASSES[0], "confidence": 0.850000},
            {"class_name": CLASSES[1], "confidence": 0.100000},
            {"class_name": CLASSES[2], "confidence": 0.050000},
        ],
        "model_name": "resnet50",
        "inference_time_ms": 42.5,
        # no "cached" key — the route pops it from the stored payload
    }


@pytest.fixture
def mock_detect_response() -> dict:
    """Stored cache payload for /detect.

    Two detections: conf 0.92 (passes threshold 0.9) and 0.45 (filtered).
    x1 < x2 and y1 < y2 validated by test_detect_bbox_coordinates_valid.
    """
    return {
        "detections": [
            {
                "class_name": "dog",
                "confidence": 0.920000,
                "x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 80.0,
            },
            {
                "class_name": "cat",
                "confidence": 0.450000,
                "x1": 60.0, "y1": 10.0, "x2": 90.0, "y2": 40.0,
            },
        ],
        "inference_time_ms": 38.2,
        # no "cached" key
    }


# ── FastAPI TestClient with mocked startup ─────────────────────────────────────

@pytest.fixture
def app_client(
    monkeypatch: pytest.MonkeyPatch,
    mock_model: MockClassificationModel,
    mock_redis: MagicMock,
) -> Generator[TestClient, None, None]:
    """TestClient whose app.state is fully mocked — no real models or Redis.

    Strategy
    --------
    1. monkeypatch api.main.load_all_models → returns ({}, {}) before lifespan runs.
    2. Enter TestClient context → lifespan runs (with no-op model loader).
    3. mock_startup() overrides app.state with test doubles.

    The eval_transform mock returns a (3, 224, 224) tensor so that
    tensor.unsqueeze(0) → (1, 3, 224, 224) feeds MockClassificationModel correctly.
    """
    from api.main import app

    # Prevent real model loading / HF Hub downloads
    monkeypatch.setattr("api.main.load_all_models", lambda device="cpu": ({}, {}))

    def mock_startup() -> None:
        app.state.models_ready = True
        app.state.models = {
            "resnet50": mock_model,
            "mobilenet": mock_model,
        }
        app.state.model_hashes = {
            "resnet50": "abcd1234",
            "mobilenet": "efgh5678",
        }
        app.state.redis = mock_redis
        # eval_transform must return a 3-D tensor so .unsqueeze(0) gives (1,C,H,W)
        app.state.eval_transform = MagicMock(
            return_value=torch.zeros(3, 224, 224)
        )
        app.state.drift_detector = MagicMock()

    # raise_server_exceptions=False: unhandled server errors become HTTP 500
    # responses rather than being re-raised in the test, so status-code assertions work.
    with TestClient(app, raise_server_exceptions=False) as client:
        mock_startup()   # override state set by the (mocked) lifespan
        yield client


# ── DriftDetector baseline fixture ────────────────────────────────────────────

@pytest.fixture
def synthetic_baseline(tmp_path: Path) -> Path:
    """Create a minimal drift baseline: 3 classes × 30 confidence scores.

    Returns the Path to the JSON metadata file (passed as baseline_path to
    DriftDetector.__init__). The .npy files sit alongside the JSON.

    Class means are slightly different (0.0, 0.1, 0.2) so distributions are
    distinguishable, while remaining within realistic confidence ranges [0, 1].
    """
    np.random.seed(42)
    class_names = ["person", "bicycle", "car"]
    metadata: dict = {
        "model_used": "mobilenet",
        "split_used": "val",
        "classes": {},
    }

    for cls_name in class_names:
        filename = f"{cls_name}_scores.npy"
        mean = class_names.index(cls_name) * 0.1  # 0.0, 0.1, 0.2
        # Confidence scores clipped to [0, 1]
        scores = np.clip(
            np.random.normal(loc=0.6 + mean, scale=0.1, size=30),
            0.0, 1.0,
        ).astype(np.float32)
        np.save(tmp_path / filename, scores)
        metadata["classes"][cls_name] = {"scores_file": filename}

    baseline_json = tmp_path / "training_confidence_baseline.json"
    baseline_json.write_text(json.dumps(metadata))
    return baseline_json


# ── fakeredis client for RedisCache unit tests ─────────────────────────────────

@pytest.fixture
def fake_redis_client():
    """FakeStrictRedis — real Redis semantics without a running server."""
    import fakeredis
    return fakeredis.FakeStrictRedis(decode_responses=False)
