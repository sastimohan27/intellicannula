"""Per-patient time series of IV-site risk, with alerting.

Trend matters more than any single reading. Absolute risk on one photo is
confounded by lighting, camera, and the patient's own skin tone; the *change*
at one site across visits cancels most of that. So the alert rules below weigh
deterioration at least as heavily as level.

Calibration caveat: the thresholds are the ones specified in the project plan,
not values fitted to outcome data - no longitudinal IV-site dataset exists.
They are starting points to be re-tuned against a real pilot, and the false
alarm / missed detection balance they imply has never been measured.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from infection_module.src.feature_extractor import extract_all_features
from infection_module.src.registration import register_images
from infection_module.src.risk_model import predict_risk

log = logging.getLogger(__name__)

RISK_THRESHOLD = 0.6
DELTA_THRESHOLD = 0.15
CONSECUTIVE_INCREASES = 3
TREND_WINDOW = 3
TREND_EPSILON = 0.02  # below this, movement is noise rather than a trend


@dataclass
class Checkin:
    """One observation of a patient's IV site."""

    timestamp: datetime
    risk_score: float
    color_distance: float
    features: dict[str, float]
    registration_degraded: bool = False


@dataclass
class PatientSiteTracker:
    """In-memory time series per patient.

    Phase 6 mirrors this to a `checkins` table; the in-memory copy is what the
    alert logic reads so a DB round trip is not on the critical path of an
    upload.
    """

    history: dict[str, list[Checkin]] = field(default_factory=lambda: defaultdict(list))
    baselines: dict[str, np.ndarray] = field(default_factory=dict)

    def set_baseline(self, patient_id: str, image: np.ndarray) -> None:
        """Record the reference image every later visit is registered against."""
        self.baselines[patient_id] = image.copy()

    def add_checkin(
        self,
        patient_id: str,
        image: np.ndarray,
        roi_bbox: list[float],
        timestamp: datetime | None = None,
    ) -> dict:
        """Register, extract features, score risk, append to the series.

        The first check-in for a patient becomes the baseline automatically, so
        a site can be enrolled and measured in one call.
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        degraded = False

        baseline = self.baselines.get(patient_id)
        if baseline is None:
            self.set_baseline(patient_id, image)
            aligned = image
        else:
            aligned, info = register_images(
                baseline, image, roi_bbox=roi_bbox, return_info=True
            )
            degraded = info["degraded"]

        features = extract_all_features(aligned, roi_bbox)
        risk = predict_risk(features)

        previous = self.history[patient_id][-1] if self.history[patient_id] else None
        delta = risk - previous.risk_score if previous else 0.0

        self.history[patient_id].append(
            Checkin(
                timestamp=timestamp,
                risk_score=risk,
                color_distance=features["color_lab_distance"],
                features=features,
                registration_degraded=degraded,
            )
        )

        return {
            "patient_id": patient_id,
            "risk_score": round(risk, 4),
            "delta_from_previous": round(delta, 4),
            "color_distance": round(features["color_lab_distance"], 3),
            "n_checkins": len(self.history[patient_id]),
            "registration_degraded": degraded,
        }

    def check_alert(self, patient_id: str) -> dict:
        """Evaluate the alert rules against a patient's series.

        Three independent triggers, ORed:

        1. `risk_score > 0.6` - the site looks abnormal right now.
        2. `delta_from_previous > 0.15` - a sharp jump since the last visit,
           which fires even when the absolute level is still moderate. This is
           the rule that catches fast deterioration early.
        3. three consecutive increases - a slow, monotone climb that never
           trips either threshold on its own. Cellulitis often progresses this
           way, and rules 1 and 2 alone would miss it until late.

        Every alert names its rule in `reason`: a clinician must be able to see
        *why* the system flagged a site to judge whether to act.
        """
        series = self.history.get(patient_id, [])
        if not series:
            return {
                "alert": False, "reason": "no check-ins recorded",
                "risk_score": 0.0, "trend": "stable",
            }

        latest = series[-1]
        risk = latest.risk_score
        delta = risk - series[-2].risk_score if len(series) >= 2 else 0.0
        trend = self._trend(series)

        reasons = []
        if risk > RISK_THRESHOLD:
            reasons.append(f"risk {risk:.2f} exceeds {RISK_THRESHOLD}")
        if delta > DELTA_THRESHOLD:
            reasons.append(f"risk rose {delta:.2f} since last check-in")
        if self._consecutive_increases(series) >= CONSECUTIVE_INCREASES:
            reasons.append(f"{CONSECUTIVE_INCREASES} consecutive increases")

        result = {
            "alert": bool(reasons),
            "reason": "; ".join(reasons) if reasons else "within normal limits",
            "risk_score": round(risk, 4),
            "delta_from_previous": round(delta, 4),
            "trend": trend,
            "n_checkins": len(series),
        }
        if latest.registration_degraded:
            # Surfaced rather than buried: a translation-only alignment means the
            # comparison against baseline is weaker than usual.
            result["caveat"] = (
                "registration fell back to translation-only - cross-visit "
                "comparison is degraded, interpret the delta with caution"
            )
        return result

    @staticmethod
    def _consecutive_increases(series: list[Checkin]) -> int:
        """Length of the current run of strictly increasing risk scores."""
        count = 0
        for earlier, later in zip(series, series[1:]):
            count = count + 1 if later.risk_score > earlier.risk_score else 0
        return count

    @staticmethod
    def _trend(series: list[Checkin]) -> str:
        """Direction over the recent window.

        Uses the slope of a least-squares fit rather than first-vs-last, so a
        single noisy reading does not flip the label. Movement smaller than
        TREND_EPSILON per visit is reported as stable.
        """
        window = [c.risk_score for c in series[-TREND_WINDOW:]]
        if len(window) < 2:
            return "stable"

        slope = float(np.polyfit(range(len(window)), window, 1)[0])
        if slope > TREND_EPSILON:
            return "worsening"
        if slope < -TREND_EPSILON:
            return "improving"
        return "stable"

    def get_series(self, patient_id: str) -> list[dict]:
        """Time series for charting in the Phase 6 dashboard."""
        return [
            {
                "timestamp": c.timestamp.isoformat(),
                "risk_score": round(c.risk_score, 4),
                "color_distance": round(c.color_distance, 3),
                "registration_degraded": c.registration_degraded,
            }
            for c in self.history.get(patient_id, [])
        ]
