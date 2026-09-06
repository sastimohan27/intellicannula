"""Tests for the alert logic.

The alert rules are what a nurse would act on, so the failure cases matter most:
a slow monotone climb that never trips a threshold must still alert, and a
degraded registration must be surfaced rather than buried.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from infection_module.src.temporal_tracker import Checkin, PatientSiteTracker


def _series(tracker, patient, scores, degraded=False):
    """Inject a risk series directly, bypassing the model."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tracker.history[patient] = [
        Checkin(
            timestamp=start + timedelta(days=i),
            risk_score=score,
            color_distance=score * 20,
            features={},
            registration_degraded=degraded,
        )
        for i, score in enumerate(scores)
    ]
    return tracker


class TestCheckAlertTriggers:
    def test_no_checkins_reports_no_alert(self):
        result = PatientSiteTracker().check_alert("nobody")
        assert result["alert"] is False
        assert "no check-ins" in result["reason"]

    def test_high_absolute_risk_alerts(self):
        t = _series(PatientSiteTracker(), "p", [0.71])
        result = t.check_alert("p")
        assert result["alert"] is True
        assert "exceeds" in result["reason"]

    def test_sharp_jump_alerts_even_below_the_risk_threshold(self):
        """The rule that catches fast deterioration early."""
        t = _series(PatientSiteTracker(), "p", [0.20, 0.45])
        result = t.check_alert("p")
        assert result["alert"] is True
        assert "rose" in result["reason"]
        assert result["risk_score"] < 0.6

    def test_three_slow_increases_alert_without_tripping_either_threshold(self):
        """Cellulitis often creeps; rules 1 and 2 alone would miss this."""
        t = _series(PatientSiteTracker(), "p", [0.20, 0.28, 0.36, 0.44])
        result = t.check_alert("p")
        assert result["alert"] is True
        assert "consecutive increases" in result["reason"]
        assert result["risk_score"] < 0.6
        assert result["delta_from_previous"] < 0.15

    def test_two_increases_are_not_enough(self):
        t = _series(PatientSiteTracker(), "p", [0.20, 0.28, 0.36])
        assert t.check_alert("p")["alert"] is False

    def test_stable_low_series_does_not_alert(self):
        t = _series(PatientSiteTracker(), "p", [0.15, 0.14, 0.16, 0.15])
        result = t.check_alert("p")
        assert result["alert"] is False
        assert result["reason"] == "within normal limits"

    def test_improving_series_does_not_alert(self):
        t = _series(PatientSiteTracker(), "p", [0.55, 0.40, 0.25, 0.12])
        assert t.check_alert("p")["alert"] is False

    def test_reason_names_every_rule_that_fired(self):
        """A clinician must see why the system flagged a site."""
        t = _series(PatientSiteTracker(), "p", [0.30, 0.50, 0.70, 0.90])
        reason = t.check_alert("p")["reason"]
        assert "exceeds" in reason and "rose" in reason and "consecutive" in reason


class TestTrend:
    def test_rising_series_is_worsening(self):
        t = _series(PatientSiteTracker(), "p", [0.10, 0.30, 0.50])
        assert t.check_alert("p")["trend"] == "worsening"

    def test_falling_series_is_improving(self):
        t = _series(PatientSiteTracker(), "p", [0.50, 0.30, 0.10])
        assert t.check_alert("p")["trend"] == "improving"

    def test_flat_series_is_stable(self):
        t = _series(PatientSiteTracker(), "p", [0.30, 0.30, 0.30])
        assert t.check_alert("p")["trend"] == "stable"

    def test_single_noisy_reading_does_not_flip_the_trend(self):
        """Slope over a window, not first-vs-last."""
        t = _series(PatientSiteTracker(), "p", [0.30, 0.31, 0.30])
        assert t.check_alert("p")["trend"] == "stable"

    def test_one_checkin_is_stable(self):
        t = _series(PatientSiteTracker(), "p", [0.42])
        assert t.check_alert("p")["trend"] == "stable"


class TestDegradedRegistration:
    def test_caveat_is_surfaced_not_buried(self):
        t = _series(PatientSiteTracker(), "p", [0.30, 0.50], degraded=True)
        assert "degraded" in t.check_alert("p")["caveat"]

    def test_no_caveat_when_registration_was_clean(self):
        t = _series(PatientSiteTracker(), "p", [0.30, 0.50], degraded=False)
        assert "caveat" not in t.check_alert("p")


class TestSeriesAccess:
    def test_get_series_is_chart_ready(self):
        t = _series(PatientSiteTracker(), "p", [0.1, 0.2])
        series = t.get_series("p")
        assert len(series) == 2
        assert set(series[0]) == {
            "timestamp", "risk_score", "color_distance", "registration_degraded"
        }

    def test_unknown_patient_returns_empty(self):
        assert PatientSiteTracker().get_series("nobody") == []

    def test_first_checkin_becomes_the_baseline(self):
        """Enrol and measure in one call."""
        tracker = PatientSiteTracker()
        image = np.full((200, 200, 3), 150, dtype=np.uint8)
        tracker.set_baseline("p", image)
        assert "p" in tracker.baselines
