# filename: api/routes/drift.py
# purpose:  GET /drift/status — per-class KS drift results for monitoring dashboards.
#           DriftDetector is injected via app.state (never imported from src/ here).
#           503 if drift detector was not initialized (missing baseline file at startup).

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class ClassDriftStatus(BaseModel):
    buffer_size: int
    min_samples_required: int = 100   # default matches config default
    tested: bool
    ks_stat: Optional[float] = None
    p_value: Optional[float] = None
    is_alert: Optional[bool] = None
    n_live: Optional[int] = None
    n_baseline: Optional[int] = None
    tested_at: Optional[float] = None
    # tested_at is a Unix epoch timestamp (UTC).
    # Convert for display: datetime.utcfromtimestamp(tested_at).isoformat()


class DriftSummary(BaseModel):
    total_classes: int
    classes_with_data: int
    classes_tested: int
    classes_alerting: int
    baseline_model: str
    baseline_split: str


class DriftStatusResponse(BaseModel):
    summary: DriftSummary
    classes: dict[str, ClassDriftStatus]


@router.get("/drift/status", response_model=DriftStatusResponse)
async def drift_status(request: Request) -> DriftStatusResponse:
    """Return per-class KS drift results.

    Classes with fewer than min_samples_required live inferences show tested=False.
    Alert fires when KS stat > 0.10 AND p < 0.05 (double-gate to prevent false positives
    from small baseline n=30 reference samples).
    """
    detector = getattr(request.app.state, "drift_detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized. Baseline file may be missing.",
        )
    return DriftStatusResponse(**detector.get_status())
