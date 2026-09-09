"""SQLAlchemy models.

Every ML prediction is stored with the model version that produced it. That is
not bookkeeping - it is what makes the audit trail usable. When a model is
retrained and behaviour changes, you have to be able to say which version
produced a recommendation a clinician acted on.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Medical record number. Unique because two records for one patient would
    # split their site history and defeat the trend analysis entirely.
    mrn: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    vein_scans: Mapped[list["VeinScan"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    checkins: Mapped[list["SiteCheckin"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("mrn", name="uq_patient_mrn"),)


class VeinScan(Base):
    """One pre-cannulation scan: image in, insertion recommendation out."""

    __tablename__ = "vein_scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )

    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    mask_url: Mapped[str | None] = mapped_column(Text)

    # Nullable throughout: the site selector legitimately refuses when no
    # segment meets the width and straightness bar. A row with a null insertion
    # point and a reason is a valid, meaningful result - not a failed request.
    insertion_x: Mapped[int | None] = mapped_column(Integer)
    insertion_y: Mapped[int | None] = mapped_column(Integer)
    angle_deg: Mapped[float | None] = mapped_column(Float)
    vein_width_px: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text)

    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="vein_scans")


class SiteCheckin(Base):
    """One post-cannulation observation of an IV site."""

    __tablename__ = "site_checkins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )

    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    roi_bbox: Mapped[str | None] = mapped_column(Text)  # JSON [x1,y1,x2,y2]

    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    delta_from_previous: Mapped[float] = mapped_column(Float, default=0.0)
    color_distance: Mapped[float | None] = mapped_column(Float)
    alert_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_reason: Mapped[str | None] = mapped_column(Text)
    trend: Mapped[str | None] = mapped_column(String(16))

    # True when registration fell back to translation-only. Persisted because a
    # clinician reading a trend later needs to know which points in it are
    # weakly aligned.
    registration_degraded: Mapped[bool] = mapped_column(Boolean, default=False)

    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="checkins")
