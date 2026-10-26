"""Pre-cannulation vein scan endpoint."""

from __future__ import annotations

import base64
import logging
import time

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import read_image_upload, require_patient
from app.core.audit import audit_ml_call
from app.core.config import get_settings
from app.core.db import get_db
from app.core.ml import ModelNotAvailable, run_vein_scan
from app.core.modality import NIRNotVerified, verify_nir
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
    x_nir_device: str | None = Header(default=None),
    x_nir_device_token: str | None = Header(default=None),
) -> VeinScanOut:
    """Segment veins, choose an insertion site, suggest a needle angle.

    A 201 with `insertion_x` null is a *successful* refusal: no vein segment met
    the minimum width and straightness bar. `reason` explains it. This is not an
    error case - forcing a recommendation the model cannot support is exactly
    what a clinical tool must not do.
    """
    data = await read_image_upload(file)
    digest = sha256_bytes(data)

    # NIR only: the model must never see an RGB photo. See app/core/modality.py.
    try:
        provenance = verify_nir(data, x_nir_device, x_nir_device_token)
    except NIRNotVerified as exc:
        log.info("vein-scan rejected for patient %s: %s", patient_id, exc.reason)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    started = time.perf_counter()
    try:
        with audit_ml_call(
            "vein-scan", patient_id, settings.vein_model_version, digest
        ) as record:
            result = run_vein_scan(data)
            inference_ms = round((time.perf_counter() - started) * 1000, 1)
            record.update({
                "insertion_x": result["insertion_x"],
                "insertion_y": result["insertion_y"],
                "confidence": result["confidence"],
                "nir_source": provenance["source"],
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
    response.mask_png_base64 = base64.b64encode(result["mask_png"]).decode("ascii")
    response.inference_ms = inference_ms
    response.modality_status = provenance["status"]
    response.acquisition_source = provenance["source"]
    response.dataset_split = provenance["dataset_split"]
    response.vein_grade = _calculate_vein_grade(result["insertion_x"], result["vein_width_px"])
    return response


@router.get("/{patient_id}/vein-scans", response_model=list[VeinScanOut])
def list_vein_scans(
    patient_id: int,
    patient: Patient = Depends(require_patient),
    db: Session = Depends(get_db),
) -> list[VeinScanOut]:
    scans = (
        db.query(VeinScan)
        .filter(VeinScan.patient_id == patient_id)
        .order_by(VeinScan.created_at.desc())
        .all()
    )
    result = []
    for scan in scans:
        out = VeinScanOut.model_validate(scan)
        out.vein_grade = _calculate_vein_grade(scan.insertion_x, scan.vein_width_px)
        result.append(out)
    return result


def _calculate_vein_grade(insertion_x: int | None, width: float | None) -> str:
    """Grade the chosen vein by its apparent width (px at 512x512).

    Deliberately not based on `confidence`: that is the selector's margin of the
    best segment over the runner-up, so an arm with several good veins scores
    *low* confidence. Grading on it marked a 13.8 px vein "Not Recommended".
    """
    if insertion_x is None or width is None:
        return "Not Recommended"
    if width > 12.0:
        return "Grade A (Excellent)"
    if width > 8.0:
        return "Grade B (Moderate)"
    return "Grade C (Difficult)"
