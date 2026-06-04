# filename: api/schemas.py
# purpose:  Pydantic request/response models shared across all routes.

from __future__ import annotations

from pydantic import BaseModel


class ClassifyPrediction(BaseModel):
    class_name: str
    confidence: float


class ClassifyResponse(BaseModel):
    predictions: list[ClassifyPrediction]
    model_name: str
    inference_time_ms: float
    cached: bool


class DetectBBox(BaseModel):
    class_name: str
    confidence: float
    # Pixel coordinates in the original input image space — NOT normalized.
    # Divide by image.width / image.height to normalize for display in Streamlit.
    x1: float
    y1: float
    x2: float
    y2: float


class DetectResponse(BaseModel):
    detections: list[DetectBBox]
    inference_time_ms: float
    cached: bool


class HealthResponse(BaseModel):
    status: str           # "ok" | "loading"
    models_ready: bool
    models_loaded: list[str]
