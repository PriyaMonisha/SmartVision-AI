# filename: api/main.py
# purpose:  FastAPI application — lifespan startup/shutdown + route registration.
#
# Rules applied:
#   Rule 32: lifespan context manager (not deprecated @app.on_event)
#   Rule 31: asyncio.get_running_loop() — event loop is already running inside async function
#   Rule 23: Redis failure handled gracefully in RedisCache.__init__ (1s timeout)
#   Rule 4:  src/ never imports from api/ — dependency flows api/ → src/ only

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

import config as cfg
from api.routes import classify, detect, health, metrics
from src.data.augmentor import get_eval_transforms
from src.inference.model_loader import load_all_models
from src.inference.redis_cache import RedisCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialise state flags before model loading so /health can respond immediately
    app.state.models_ready = False
    app.state.models = {}
    app.state.model_hashes = {}

    # Rule 31: get_running_loop() — loop is already running in an async context.
    # load_all_models is blocking (torch.load, HF Hub HTTP); run in thread pool so
    # the event loop stays responsive during startup (/health returns 503 while loading).
    loop = asyncio.get_running_loop()
    try:
        models, hashes = await loop.run_in_executor(
            None, lambda: load_all_models(device="cpu")
        )
        app.state.models = models
        app.state.model_hashes = hashes
        app.state.models_ready = True
        logger.info(f"Models ready: {list(models.keys())}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        # models_ready stays False — /health returns 503 on every request

    # Redis — 1s socket_connect_timeout; disables cache gracefully if Redis is down
    app.state.redis = RedisCache(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)

    # Build eval transform once at startup — not per-request (avoids per-call allocation)
    app.state.eval_transform = get_eval_transforms(image_size=cfg.IMAGE_SIZE)

    yield

    # Shutdown cleanup
    app.state.models = {}
    app.state.model_hashes = {}
    app.state.models_ready = False
    logger.info("SmartVision API shutdown complete.")


app = FastAPI(
    title="SmartVision AI",
    description=(
        "Multi-class object recognition (ResNet50, MobileNetV2) "
        "and detection (YOLOv8n) API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(classify.router)
app.include_router(detect.router)
app.include_router(metrics.router)
