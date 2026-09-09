"""Risk history for the trend chart."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_patient
from app.core.db import get_db
from app.models import Patient, SiteCheckin
from app.schemas import HistoryPoint, PatientHistory

router = APIRouter(prefix="/patients", tags=["history"])


@router.get("/{patient_id}/history", response_model=PatientHistory)
def get_history(
    patient_id: int,
    patient: Patient = Depends(require_patient),
    db: Session = Depends(get_db),
) -> PatientHistory:
    """Full check-in series, oldest first.

    Ascending order because this feeds a time-series chart; returning newest
    first would require the client to reverse it and invites an off-by-one in
    the delta rendering.
    """
    checkins = (
        db.query(SiteCheckin)
        .filter(SiteCheckin.patient_id == patient_id)
        .order_by(SiteCheckin.created_at.asc())
        .all()
    )

    return PatientHistory(
        patient_id=patient_id,
        n_checkins=len(checkins),
        latest_trend=checkins[-1].trend if checkins else None,
        points=[
            HistoryPoint(
                timestamp=c.created_at,
                risk_score=c.risk_score,
                delta_from_previous=c.delta_from_previous,
                color_distance=c.color_distance,
                alert_triggered=c.alert_triggered,
                registration_degraded=c.registration_degraded,
            )
            for c in checkins
        ],
    )
