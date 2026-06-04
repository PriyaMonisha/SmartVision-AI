# filename: api/routes/classify.py
# purpose:  POST /classify — CNN image classification with Redis cache-aside.
#           Image input via UploadFile (binary); model_name/top_k via Form (text).
#           Using Form() for binary data causes 422 Unprocessable Entity.

from __future__ import annotations

import io
import time
from typing import Annotated

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

import config as cfg
from api.prometheus_metrics import (
    cache_hits,
    cache_misses,
    classify_latency,
    classify_requests,
)
from api.schemas import ClassifyPrediction, ClassifyResponse
from src.inference.redis_cache import RedisCache

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(
    request: Request,
    file: Annotated[UploadFile, File(description="Image file to classify")],
    model_name: str = Form(default="resnet50"),
    top_k: int = Form(default=5),
) -> ClassifyResponse:
    if not getattr(request.app.state, "models_ready", False):
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    models = request.app.state.models
    if model_name not in models or model_name == "yolo":
        available = [k for k in models if k != "yolo"]
        raise HTTPException(
            status_code=400,
            detail=f"model_name must be one of: {available}",
        )

    image_bytes = await file.read()

    # Cache lookup
    model_hash = request.app.state.model_hashes.get(model_name, "")
    cache: RedisCache = request.app.state.redis
    cache_key = RedisCache.make_classify_key(image_bytes, model_name, model_hash)

    cached_data = cache.get(cache_key)
    if cached_data is not None:
        cache_hits.labels(endpoint="classify").inc()
        # Pop "cached" before unpacking — stored dict has cached=False which
        # conflicts with the explicit cached=True kwarg (TypeError: keyword repeated).
        cached_data.pop("cached", None)
        return ClassifyResponse(**cached_data, cached=True)

    cache_misses.labels(endpoint="classify").inc()

    # Preprocess — eval_transform built once at startup, stored in app.state
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = request.app.state.eval_transform(image).unsqueeze(0)

    # Inference
    model = models[model_name]
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    # Top-k predictions
    k = min(top_k, len(cfg.CLASSES))
    top_probs, top_idxs = probs.topk(k)
    predictions = [
        ClassifyPrediction(
            class_name=cfg.CLASSES[idx.item()],
            confidence=round(prob.item(), 6),
        )
        for prob, idx in zip(top_probs, top_idxs)
    ]

    # Prometheus — guard class_name to prevent unbounded cardinality
    top1 = predictions[0].class_name
    if top1 not in cfg.CLASSES:
        top1 = "unknown"
    classify_requests.labels(class_name=top1, model_name=model_name).inc()
    classify_latency.observe(inference_ms / 1000)

    result = ClassifyResponse(
        predictions=predictions,
        model_name=model_name,
        inference_time_ms=round(inference_ms, 2),
        cached=False,
    )

    # Cache without "cached" field — popped on retrieval above
    payload = result.model_dump()
    payload.pop("cached", None)
    cache.set(cache_key, payload, ttl=cfg.REDIS_TTL_CLASSIFY)

    return result
