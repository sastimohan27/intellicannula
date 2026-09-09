"""Pydantic request/response models.

These are the contract the frontend types mirror. Field names match the ORM
models so a response can be built straight from a row.
"""

from __future__ import annotations

from datetime import datetime

import json

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    mrn: str = Field(min_length=1, max_length=64)


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    mrn: str
    created_at: datetime


class VeinScanOut(BaseModel):
    """Result of a pre-cannulation scan.

    `insertion_x/y` are null when the selector refused - that is a valid
    outcome, not an error, and `reason` says why. Clients must handle it:
    rendering a marker at (0, 0) because the field was null would put a
    recommendation on the corner of the image.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: int
    image_url: str
    mask_url: str | None = None
    insertion_x: int | None = None
    insertion_y: int | None = None
    angle_deg: float | None = None
    vein_width_px: float | None = None
    confidence: float
    reason: str | None = None
    model_version: str
    created_at: datetime

    # Base64 PNG of the original with mask and marker drawn on. Not persisted -
    # it is regenerated per request so it always matches the stored numbers.
    overlay_png_base64: str | None = None


class SiteCheckinOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: int
    image_url: str
    roi_bbox: list[float] | None = None

    @field_validator("roi_bbox", mode="before")
    @classmethod
    def _parse_roi_bbox(cls, value):
        """Accept the JSON string the database column holds.

        roi_bbox is persisted as Text (portable across SQLite and Postgres
        without a JSON column type), so validating straight from an ORM row
        hands this field a string. Parsing here keeps the wire format a real
        list instead of leaking storage representation to the client.
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value
    risk_score: float
    delta_from_previous: float
    color_distance: float | None = None
    alert_triggered: bool
    alert_reason: str | None = None
    trend: str | None = None
    registration_degraded: bool
    model_version: str
    created_at: datetime


class HistoryPoint(BaseModel):
    """One point on the risk trend chart."""

    timestamp: datetime
    risk_score: float
    delta_from_previous: float
    color_distance: float | None = None
    alert_triggered: bool
    registration_degraded: bool


class PatientHistory(BaseModel):
    patient_id: int
    n_checkins: int
    points: list[HistoryPoint]
    latest_trend: str | None = None


class AlertMessage(BaseModel):
    """Pushed over the /ws/alerts WebSocket."""

    patient_id: int
    patient_name: str
    risk_score: float
    reason: str
    trend: str
    created_at: datetime


class ErrorDetail(BaseModel):
    detail: str
