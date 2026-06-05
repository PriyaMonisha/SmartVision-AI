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

from fastapi import FastAPI, Request, Response

import config as cfg
from api.prometheus_metrics import http_errors, models_loaded
from api.routes import classify, detect, drift, ensemble, health, metrics
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
        models_loaded.set(1)
        logger.info(f"Models ready: {list(models.keys())}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        # models_ready stays False — /health returns 503 on every request

    # Redis — 1s socket_connect_timeout; disables cache gracefully if Redis is down
    app.state.redis = RedisCache(
        host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, password=cfg.REDIS_PASSWORD
    )

    # Build eval transform once at startup — not per-request (avoids per-call allocation)
    app.state.eval_transform = get_eval_transforms(image_size=cfg.IMAGE_SIZE)

    # Drift detector — initialized after Redis so redis_client is available for buffer restore
    from src.monitoring.drift_detector import DriftDetector

    try:
        app.state.drift_detector = DriftDetector(
            baseline_path=cfg.DRIFT_BASELINE_PATH,
            redis_client=app.state.redis,
            min_samples=cfg.KS_MIN_LIVE_SAMPLES,
            alert_threshold=cfg.KS_DRIFT_ALERT_THRESHOLD,
        )
        logger.info("DriftDetector initialized: 22-class KS monitoring active")
    except FileNotFoundError as e:
        logger.error(f"DriftDetector init failed (baseline missing): {e}")
        app.state.drift_detector = None

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


@app.middleware("http")
async def count_http_errors(request: Request, call_next) -> Response:
    """Increment http_errors counter for any 4xx/5xx response."""
    response = await call_next(request)
    if response.status_code >= 400:
        endpoint = request.url.path
        http_errors.labels(
            status_code=str(response.status_code),
            endpoint=endpoint,
        ).inc()
    return response


app.include_router(health.router)
app.include_router(classify.router)
app.include_router(ensemble.router)
app.include_router(detect.router)
app.include_router(drift.router)
app.include_router(metrics.router)
