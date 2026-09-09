"""Patient CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Patient
from app.schemas import PatientCreate, PatientOut

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("", response_model=PatientOut, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)) -> Patient:
    patient = Patient(name=payload.name, mrn=payload.mrn)
    db.add(patient)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # 409, not 400: the request is well-formed, it conflicts with existing
        # state. Duplicate MRNs would split a patient's history across records.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a patient with MRN {payload.mrn!r} already exists",
        ) from None
    db.refresh(patient)
    return patient


@router.get("", response_model=list[PatientOut])
def list_patients(db: Session = Depends(get_db), limit: int = 50) -> list[Patient]:
    return db.query(Patient).order_by(Patient.created_at.desc()).limit(limit).all()


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: int, db: Session = Depends(get_db)) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"patient {patient_id} not found")
    return patient
