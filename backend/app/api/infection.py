"""Post-cannulation site-check endpoint."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.alerts import broadcast_alert
from app.api.deps import read_image_upload, require_patient
from app.core.audit import audit_ml_call
from app.core.config import get_settings
from app.core.db import get_db
from app.core.ml import ModelNotAvailable, run_site_check
from app.core.storage import get_store, sha256_bytes
from app.models import Patient, SiteCheckin
from app.schemas import AlertMessage, SiteCheckinOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/patients", tags=["infection"])
settings = get_settings()

RISK_THRESHOLD = 0.6
DELTA_THRESHOLD = 0.15
CONSECUTIVE_INCREASES = 3


@router.post("/{patient_id}/site-check", response_model=SiteCheckinOut, status_code=201)
async def create_site_check(
    patient_id: int,
    file: UploadFile = File(...),
    patient: Patient = Depends(require_patient),
    db: Session = Depends(get_db),
) -> SiteCheckin:
    """Score an IV site photo against this patient's own history.

    The first check-in becomes the baseline every later one is registered
    against, so a patient can be enrolled and measured in one call.
    """
    data = await read_image_upload(file)
    digest = sha256_bytes(data)
    store = get_store()

    previous = (
        db.query(SiteCheckin)
        .filter(SiteCheckin.patient_id == patient_id)
        .order_by(SiteCheckin.created_at.asc())
        .all()
    )
    baseline_bytes = store.load(previous[0].image_url) if previous else None

    try:
        with audit_ml_call(
            "site-check", patient_id, settings.infection_model_version, digest
        ) as record:
            result = run_site_check(data, patient_id, baseline_bytes)
            record.update({
                "risk_score": result["risk_score"],
                "registration_degraded": result["registration_degraded"],
            })
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    risk = result["risk_score"]
    delta = risk - previous[-1].risk_score if previous else 0.0
    history = [c.risk_score for c in previous] + [risk]
    alert, reason = _evaluate_alert(risk, delta, history)

    checkin = SiteCheckin(
        patient_id=patient_id,
        image_url=store.save(data, patient_id, "site", ".png"),
        roi_bbox=json.dumps(result["roi_bbox"]),
        risk_score=risk,
        delta_from_previous=delta,
        color_distance=result["color_distance"],
        alert_triggered=alert,
        alert_reason=reason,
        trend=_trend(history),
        registration_degraded=result["registration_degraded"],
        model_version=result["model_version"],
        image_sha256=digest,
    )
    db.add(checkin)
    db.commit()
    db.refresh(checkin)

    if alert:
        await broadcast_alert(
            AlertMessage(
                patient_id=patient_id,
                patient_name=patient.name,
                risk_score=round(risk, 4),
                reason=reason,
                trend=checkin.trend or "stable",
                created_at=checkin.created_at,
            )
        )

    response = SiteCheckinOut.model_validate(checkin)
    response.roi_bbox = result["roi_bbox"]
    response.cannula_confidence = result.get("cannula_confidence")
    redness = result.get("redness")
    if redness is not None:
        response.redness_level = redness["redness_level"]
        response.redness_label = redness["redness_label"]
        response.reddened_fraction = redness["reddened_fraction"]
        response.spread_hub_widths = redness["spread_hub_widths"]
        response.skin_visible_fraction = redness["skin_fraction"]
    response.baseline_reddened_fraction = result.get("baseline_reddened_fraction")
    response.change_since_baseline = result.get("change_since_baseline")
    response.overlay_png_base64 = result.get("overlay_png_base64")
    response.risk_velocity = _format_risk_velocity(delta)
    response.clinical_explanation = _explain(
        redness, result.get("change_since_baseline"), is_baseline=not previous,
        trend=checkin.trend,
    )
    return response


def _evaluate_alert(risk: float, delta: float, history: list[float]) -> tuple[bool, str]:
    """Three independent triggers, ORed. Mirrors PatientSiteTracker.check_alert.

    Every alert names the rule that fired: a clinician has to be able to see
    *why* a site was flagged to judge whether to act on it.
    """
    reasons = []
    if risk > RISK_THRESHOLD:
        reasons.append(f"risk {risk:.2f} exceeds {RISK_THRESHOLD}")
    if delta > DELTA_THRESHOLD:
        reasons.append(f"risk rose {delta:.2f} since last check-in")

    run = 0
    for earlier, later in zip(history, history[1:]):
        run = run + 1 if later > earlier else 0
    if run >= CONSECUTIVE_INCREASES:
        reasons.append(f"{CONSECUTIVE_INCREASES} consecutive increases")

    return bool(reasons), "; ".join(reasons) if reasons else "within normal limits"


def _trend(history: list[float]) -> str:
    """Slope over the recent window, so one noisy reading cannot flip the label."""
    import numpy as np

    window = history[-3:]
    if len(window) < 2:
        return "stable"
    slope = float(np.polyfit(range(len(window)), window, 1)[0])
    if slope > 0.02:
        return "worsening"
    if slope < -0.02:
        return "improving"
    return "stable"


def _format_risk_velocity(delta: float) -> str:
    """Format the change in risk as a clinical trend string."""
    if abs(delta) < 0.05:
        return "Stable"
    direction = "+" if delta > 0 else ""
    return f"{direction}{delta:.2f} / day"


def _explain(
    redness: dict | None, change: float | None, is_baseline: bool, trend: str | None
) -> list[str]:
    """Plain statements of what was measured - every number traceable to the photo."""
    if redness is None:
        return []
    lines = [
        f"{redness['reddened_fraction']:.0%} of the visible skin around the cannula is "
        "redder than this patient's skin further from the site "
        "(CIELAB \u0394a* above 6).",
    ]
    if redness["reddened_fraction"] >= 0.10:
        lines.append(
            f"Reddened skin reaches about {redness['spread_hub_widths']:.1f} hub widths "
            "from the insertion point."
        )
    if is_baseline:
        lines.append(
            "This is the baseline photo. Later check-ins are compared with it, so "
            "naturally redder skin (e.g. on the hand) is not mistaken for change."
        )
    elif change is None:
        lines.append("The baseline photo has no detectable cannula, so change since "
                     "insertion cannot be computed.")
    else:
        lines.append(f"Change since baseline: {change * 100:+.0f} percentage points of reddened skin.")
    if trend == "worsening":
        lines.append("Redness is rising across recent check-ins.")
    lines.append(
        f"Only {redness['skin_fraction']:.0%} of the ring around the hub is visible skin; "
        "dressing, tape and plastic are excluded from the measurement."
        if redness["skin_fraction"] < 0.4 else
        "Dressing, tape and the plastic hub are excluded from the measurement."
    )
    return lines
