"""Pre-cannulation vein scan endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import read_image_upload, require_patient
from app.core.audit import audit_ml_call
from app.core.config import get_settings
from app.core.db import get_db
from app.core.ml import ModelNotAvailable, run_vein_scan
from app.core.storage import get_store, sha256_bytes
from app.models import Patient, VeinScan
from app.schemas import VeinScanOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/patients", tags=["vein"])
settings = get_settings()


@router.post("/{patient_id}/vein-scan", response_model=VeinScanOut, status_code=201)
async def create_vein_scan(
    patient_id: int,
    file: UploadFile = File(...),
    patient: Patient = Depends(require_patient),
    db: Session = Depends(get_db),
) -> VeinScanOut:
    """Segment veins, choose an insertion site, suggest a needle angle.

    A 201 with `insertion_x` null is a *successful* refusal: no vein segment met
    the minimum width and straightness bar. `reason` explains it. This is not an
    error case - forcing a recommendation the model cannot support is exactly
    what a clinical tool must not do.
    """
    data = await read_image_upload(file)
    digest = sha256_bytes(data)

    try:
        with audit_ml_call(
            "vein-scan", patient_id, settings.vein_model_version, digest
        ) as record:
            result = run_vein_scan(data)
            record.update({
                "insertion_x": result["insertion_x"],
                "insertion_y": result["insertion_y"],
                "confidence": result["confidence"],
            })
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    store = get_store()
    image_url = store.save(data, patient_id, "vein", ".png")
    mask_url = store.save(result["mask_png"], patient_id, "vein_mask", ".png")

    scan = VeinScan(
        patient_id=patient_id,
        image_url=image_url,
        mask_url=mask_url,
        insertion_x=result["insertion_x"],
        insertion_y=result["insertion_y"],
        angle_deg=result["angle_deg"],
        vein_width_px=result["vein_width_px"],
        confidence=result["confidence"],
        reason=result["reason"],
        model_version=result["model_version"],
        image_sha256=digest,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    response = VeinScanOut.model_validate(scan)
    response.overlay_png_base64 = result["overlay_png_base64"]
    return response


@router.get("/{patient_id}/vein-scans", response_model=list[VeinScanOut])
def list_vein_scans(
    patient_id: int,
    patient: Patient = Depends(require_patient),
    db: Session = Depends(get_db),
) -> list[VeinScan]:
    return (
        db.query(VeinScan)
        .filter(VeinScan.patient_id == patient_id)
        .order_by(VeinScan.created_at.desc())
        .all()
    )
