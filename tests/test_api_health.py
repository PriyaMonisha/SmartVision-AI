"""
tests/test_api_health.py
────────────────────────
Tests for GET /health endpoint (8 tests).

Validates:
  - 200 OK when models ready; 503 when loading
  - Response body fields (status string, models_ready bool)
  - models_loaded list accuracy
  - Schema compliance (HealthResponse Pydantic model)
  - Content-Type header
  - Response latency (<100 ms — health checks must be fast)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from api.schemas import HealthResponse

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_health_returns_200_when_ready(app_client: "TestClient") -> None:
    response = app_client.get("/health")
    assert response.status_code == 200


def test_health_status_ok_when_ready(app_client: "TestClient") -> None:
    response = app_client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_ready"] is True


def test_health_returns_503_when_not_ready(app_client: "TestClient") -> None:
    app_client.app.state.models_ready = False
    try:
        response = app_client.get("/health")
        assert response.status_code == 503
    finally:
        app_client.app.state.models_ready = True  # restore for isolation


def test_health_status_loading_when_not_ready(app_client: "TestClient") -> None:
    app_client.app.state.models_ready = False
    try:
        response = app_client.get("/health")
        data = response.json()
        assert data["status"] == "loading"
        assert data["models_ready"] is False
    finally:
        app_client.app.state.models_ready = True


def test_health_models_loaded_matches_state(app_client: "TestClient") -> None:
    response = app_client.get("/health")
    data = response.json()
    expected = set(app_client.app.state.models.keys())
    assert set(data["models_loaded"]) == expected


def test_health_response_validates_schema(app_client: "TestClient") -> None:
    response = app_client.get("/health")
    data = response.json()
    # Raises ValidationError if any field is missing or wrong type
    parsed = HealthResponse(**data)
    assert parsed.models_ready is True


def test_health_content_type_is_json(app_client: "TestClient") -> None:
    response = app_client.get("/health")
    assert "application/json" in response.headers["content-type"]


def test_health_response_time_under_100ms(app_client: "TestClient") -> None:
    t0 = time.perf_counter()
    response = app_client.get("/health")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert response.status_code == 200
    assert elapsed_ms < 100, f"Health check took {elapsed_ms:.1f} ms (limit 100 ms)"
