# filename: api/routes/detect.py
# purpose:  POST /detect — YOLOv8n object detection with Redis cache-aside.
#           Uses results[0].names for class names (embedded in .pt, not config.CLASSES).
#           Image converted to numpy HWC RGB uint8 — version-stable YOLO input path.

from __future__ import annotations

import io
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, ImageOps

import config as cfg
from api.prometheus_metrics import (
    cache_hits,
    cache_misses,
    detect_latency,
    detect_requests,
)
from api.schemas import DetectBBox, DetectResponse
from src.inference.redis_cache import RedisCache

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
async def detect_objects(
    request: Request,
    file: Annotated[UploadFile, File(description="Image file for object detection")],
    conf_threshold: float = Form(default=cfg.YOLO_CONF_THRESHOLD),
) -> DetectResponse:
    if not getattr(request.app.state, "models_ready", False):
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    # Explicit None check — YOLO may have failed to load gracefully
    yolo = request.app.state.models.get("yolo")
    if yolo is None:
        raise HTTPException(
            status_code=503, detail="Detection model not loaded. Check server logs."
        )

    image_bytes = await file.read()

    cache: RedisCache = request.app.state.redis
    cache_key = RedisCache.make_detect_key(image_bytes, conf_threshold)

    cached_data = cache.get(cache_key)
    if cached_data is not None:
        cache_hits.labels(endpoint="detect").inc()
        cached_data.pop("cached", None)
        return DetectResponse(**cached_data, cached=True)

    cache_misses.labels(endpoint="detect").inc()

    # PIL → numpy HWC RGB uint8 — version-stable YOLO input (not raw PIL).
    # EXIF transpose ensures phone photos are correctly oriented before YOLO inference,
    # so returned bbox coordinates match the orientation seen by the Streamlit display.
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_image = ImageOps.exif_transpose(pil_image)
    img_array = np.array(pil_image)

    t0 = time.perf_counter()
    results = yolo.predict(img_array, conf=conf_threshold, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000

    # Use class names from the .pt file itself — not config.CLASSES
    class_names: dict[int, str] = results[0].names
    detections = [
        DetectBBox(
            class_name=class_names.get(int(box.cls[0].item()), "unknown"),
            confidence=round(float(box.conf[0].item()), 6),
            x1=round(float(box.xyxy[0][0]), 2),
            y1=round(float(box.xyxy[0][1]), 2),
            x2=round(float(box.xyxy[0][2]), 2),
            y2=round(float(box.xyxy[0][3]), 2),
        )
        for box in results[0].boxes
    ]

    detect_requests.inc()
    detect_latency.observe(inference_ms / 1000)

    result = DetectResponse(
        detections=detections,
        inference_time_ms=round(inference_ms, 2),
        cached=False,
    )

    payload = result.model_dump()
    payload.pop("cached", None)
    cache.set(cache_key, payload, ttl=cfg.REDIS_TTL_DETECT)

    return result
