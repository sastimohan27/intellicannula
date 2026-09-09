"""Shared endpoint dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Patient

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


def require_patient(patient_id: int, db: Session = Depends(get_db)) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"patient {patient_id} not found")
    return patient


async def read_image_upload(file: UploadFile) -> bytes:
    """Validate and read an uploaded image.

    422 for a bad image rather than 400: the request shape is right, the
    content is not processable. Size is checked after reading because
    content-length can be absent on streamed uploads and cannot be trusted.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"unsupported content type {file.content_type!r}; "
                f"expected one of {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    return data
