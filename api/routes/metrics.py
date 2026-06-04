# filename: api/routes/metrics.py
# purpose:  GET /metrics — Prometheus exposition format.
#           Rule 30: must return CONTENT_TYPE_LATEST (text/plain), NOT application/json.
#           FastAPI defaults to JSON; returning text/plain requires explicit Response.
#           Prometheus scraper cannot parse JSON — scrape would silently fail.

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
