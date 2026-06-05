# filename: api/routes/ensemble.py
# purpose:  POST /ensemble — weighted-average ensemble of ResNet50 + EfficientNetB0 + MobileNetV2.
#           VGG16 excluded (weights unavailable). Weights proportional to test accuracy.
#           Cache key uses all 3 model hashes so invalidation is correct on any weight update.

from __future__ import annotations

import io
import time
from typing import Annotated

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

import config as cfg
from api.schemas import ClassifyPrediction, EnsembleResponse
from src.inference.redis_cache import RedisCache

router = APIRouter()

# Weights proportional to test accuracy (ResNet50=65.5%, Efficientnet=58.9%, MobileNet=56.7%)
_ENSEMBLE_MODELS = ["resnet50", "efficientnet", "mobilenet"]
_RAW_WEIGHTS     = {"resnet50": 65.5, "efficientnet": 58.9, "mobilenet": 56.7}
_TOTAL           = sum(_RAW_WEIGHTS.values())
ENSEMBLE_WEIGHTS = {k: v / _TOTAL for k, v in _RAW_WEIGHTS.items()}


@router.post("/ensemble", response_model=EnsembleResponse)
async def ensemble_classify(
    request: Request,
    file: Annotated[UploadFile, File(description="Image file to classify")],
    top_k: int = Form(default=5),
) -> EnsembleResponse:
    if not getattr(request.app.state, "models_ready", False):
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    models = request.app.state.models
    available = [m for m in _ENSEMBLE_MODELS if m in models]
    if not available:
        raise HTTPException(status_code=503, detail="No CNN ensemble models loaded")

    image_bytes = await file.read()

    # Cache key combines all loaded model hashes
    hashes      = request.app.state.model_hashes
    combo_hash  = "_".join(hashes.get(m, "") for m in available)
    cache: RedisCache = request.app.state.redis
    cache_key   = RedisCache.make_classify_key(image_bytes, f"ensemble_{combo_hash}", "")

    cached_data = cache.get(cache_key)
    if cached_data is not None:
        cached_data.pop("cached", None)
        return EnsembleResponse(**cached_data, cached=True)

    image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = request.app.state.eval_transform(image).unsqueeze(0)

    # Run each model and collect weighted softmax
    t0           = time.perf_counter()
    per_model: dict[str, list[ClassifyPrediction]] = {}
    weighted_sum = torch.zeros(cfg.NUM_CLASSES)

    for model_name in available:
        model = models[model_name]
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]

        w = ENSEMBLE_WEIGHTS.get(model_name, 1.0 / len(available))
        weighted_sum += w * probs

        k    = min(top_k, cfg.NUM_CLASSES)
        top_probs, top_idxs = probs.topk(k)
        per_model[model_name] = [
            ClassifyPrediction(
                class_name=cfg.CLASSES[idx.item()],
                confidence=round(prob.item(), 6),
            )
            for prob, idx in zip(top_probs, top_idxs)
        ]

    total_ms = (time.perf_counter() - t0) * 1000

    # Ensemble top-k from weighted average
    k = min(top_k, cfg.NUM_CLASSES)
    top_probs, top_idxs = weighted_sum.topk(k)
    ensemble_preds = [
        ClassifyPrediction(
            class_name=cfg.CLASSES[idx.item()],
            confidence=round(prob.item(), 6),
        )
        for prob, idx in zip(top_probs, top_idxs)
    ]

    result = EnsembleResponse(
        ensemble_predictions=ensemble_preds,
        per_model=per_model,
        models_used=available,
        weights={m: round(ENSEMBLE_WEIGHTS[m], 4) for m in available},
        inference_time_ms=round(total_ms, 2),
        cached=False,
    )

    payload = result.model_dump()
    payload.pop("cached", None)
    cache.set(cache_key, payload, ttl=cfg.REDIS_TTL_CLASSIFY)

    return result
