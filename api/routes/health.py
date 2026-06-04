# filename: api/routes/health.py
# purpose:  GET /health — reports model loading status.
#           Returns 200 when models_ready=True, 503 when loading.
#           503 (not connection error) lets Docker healthcheck, Prometheus blackbox,
#           and load balancers wait until the server is ready to serve requests.

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> JSONResponse:
    ready = getattr(request.app.state, "models_ready", False)
    body = HealthResponse(
        status="ok" if ready else "loading",
        models_ready=ready,
        models_loaded=list(getattr(request.app.state, "models", {}).keys()),
    )
    # 503 during startup is correct — endpoint is reachable but not yet ready.
    # Docker HEALTHCHECK uses curl -f which treats 5xx as failure → waits.
    return JSONResponse(content=body.model_dump(), status_code=200 if ready else 503)
