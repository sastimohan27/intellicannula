"""Site monitoring: real-pipeline checks on the demo set, plus API contract cases.

The real-pipeline tests run the actual cannula detector, redness measurement
and scene check. They skip when the weights or the demo images are absent
(build the images with scripts/build_demo_set.py).
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from app.core import ml

DEMO = ml.settings.repo_root / "data/demo"
WEIGHTS = ml.settings.repo_root / "infection_module/models/cannula_yolo11.pt"

needs_pipeline = pytest.mark.skipif(
    not (WEIGHTS.exists() and (DEMO / "SIMULATED_day1_baseline.png").exists()),
    reason="cannula weights or demo images not present",
)


def _read(name: str) -> bytes:
    return (DEMO / name).read_bytes()


def _png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


@needs_pipeline
class TestRealPipeline:
    def test_baseline_then_follow_up_is_compared(self):
        base = _read("SIMULATED_day1_baseline.png")
        first = ml.run_site_check(base, 0, None)
        assert first["comparison_status"] == "baseline"
        assert first["sample_origin"] == "simulated_demo"

        later = ml.run_site_check(_read("SIMULATED_day3_spreading.png"), 0, base)
        assert later["comparison_status"] == "compared"
        assert later["scene_match_inliers"] >= ml.settings.scene_min_inliers
        assert later["change_since_baseline"] > 0.3
        assert later["baseline_overlay_png_base64"]

    def test_different_scene_is_not_compared(self):
        other = ml.settings.repo_root / "data/cannula/test/images/pt022_4.jpg"
        if not other.exists():
            pytest.skip("cannula test images not present")
        with pytest.raises(ml.SceneMismatch, match="Unable to compare with baseline"):
            ml.run_site_check(other.read_bytes(), 0, _read("SIMULATED_day1_baseline.png"))

    def test_blurred_follow_up_rejected_before_scoring(self):
        day2 = cv2.imdecode(np.frombuffer(_read("SIMULATED_day2_mild.png"), np.uint8), 1)
        with pytest.raises(ml.PhotoQualityError, match="blurred"):
            ml.run_site_check(_png(cv2.GaussianBlur(day2, (0, 0), 8)), 0,
                              _read("SIMULATED_day1_baseline.png"))

    def test_no_cannula_photo_rejected(self):
        rng = np.random.default_rng(0)
        scene = cv2.GaussianBlur(rng.integers(0, 255, (600, 800, 3), dtype=np.uint8), (0, 0), 1)
        with pytest.raises(ValueError, match="cannula"):
            ml.run_site_check(_png(scene), 0, None)

    def test_renamed_file_keeps_simulated_label(self):
        """The marker is inside the PNG, so renaming cannot hide that it is simulated."""
        assert ml.sample_origin(_read("SIMULATED_day2_mild.png")) == "simulated_demo"
        assert ml.sample_origin(_png(np.zeros((8, 8, 3), np.uint8))) == "uploaded_photo"


class TestCannulaStates:
    def _patch(self, monkeypatch, confidence):
        from infection_module.src.cannula_detector import inference

        monkeypatch.setattr(inference, "detect_cannula", lambda *_a, **_k: (
            None if confidence is None
            else {"bbox": [10.0, 10.0, 50.0, 40.0], "confidence": confidence}))
        monkeypatch.setattr(ml.settings, "repo_root", ml.settings.repo_root)

    def test_nothing_found_is_no_cannula(self, monkeypatch):
        self._patch(monkeypatch, None)
        with pytest.raises(ml.NoCannulaDetected):
            ml._locate_roi(np.zeros((300, 300, 3), np.uint8))

    def test_weak_detection_is_unable_to_verify(self, monkeypatch):
        self._patch(monkeypatch, 0.31)
        with pytest.raises(ml.CannulaUnverified, match="Unable to verify the cannula.*0.31"):
            ml._locate_roi(np.zeros((300, 300, 3), np.uint8))

    def test_threshold_is_inclusive(self, monkeypatch):
        self._patch(monkeypatch, 0.50)
        roi, confidence, hub = ml._locate_roi(np.zeros((300, 300, 3), np.uint8))
        assert confidence == 0.50
        assert hub == [10.0, 10.0, 50.0, 40.0]
        assert roi[0] < hub[0] and roi[2] > hub[2]   # margin added, clipped to frame


class TestAlertBoundaries:
    def test_risk_threshold_is_strict(self):
        from app.api.infection import _evaluate_alert

        assert _evaluate_alert(0.60, 0.0, [0.60])[0] is False
        assert _evaluate_alert(0.61, 0.0, [0.61])[0] is True

    def test_delta_threshold_is_strict(self):
        from app.api.infection import _evaluate_alert

        assert _evaluate_alert(0.30, 0.15, [0.15, 0.30])[0] is False
        assert _evaluate_alert(0.31, 0.16, [0.15, 0.31])[0] is True

    def test_assessment_statuses(self):
        from app.api.infection import _assessment

        level = {"redness_level": 0}
        assert _assessment("baseline", None, level, False)[0] == "baseline_recorded"
        assert _assessment("baseline_unusable", None, level, False)[0] == "unable_to_compare"
        assert _assessment("compared", 0.05, level, False)[0] == "no_significant_change"
        assert _assessment("compared", 0.10, level, False)[0] == "visible_change"
        assert _assessment("compared", 0.50, level, True)[0] == "review_recommended"


class TestTimelineIntegrity:
    def test_patients_do_not_share_baselines(self, client, png_bytes, mock_site_check):
        a = client.post("/patients", json={"name": "A", "mrn": "DEMO-A"}).json()
        b = client.post("/patients", json={"name": "B", "mrn": "DEMO-B"}).json()
        mock_site_check.extend([0.2, 0.9])
        client.post(f"/patients/{a['id']}/site-check", files={"file": ("x.png", png_bytes, "image/png")})
        first_b = client.post(f"/patients/{b['id']}/site-check",
                              files={"file": ("x.png", png_bytes, "image/png")}).json()
        assert first_b["delta_from_previous"] == 0.0      # B's first check-in, not compared to A
        assert first_b["assessment_status"] in {"baseline_recorded", "review_recommended"}
        assert client.get(f"/patients/{a['id']}/history").json()["n_checkins"] == 1

    def test_timestamps_are_utc_and_ordered(self, client, patient, png_bytes, mock_site_check):
        mock_site_check.extend([0.1, 0.2, 0.3])
        for _ in range(3):
            client.post(f"/patients/{patient['id']}/site-check",
                        files={"file": ("x.png", png_bytes, "image/png")})
        points = client.get(f"/patients/{patient['id']}/history").json()["points"]
        stamps = [p["timestamp"] for p in points]
        assert all(s.endswith("Z") or s.endswith("+00:00") for s in stamps)
        assert stamps == sorted(stamps)
        assert [p["risk_score"] for p in points] == [0.1, 0.2, 0.3]

    def test_baseline_reference_is_utc(self, client, patient, png_bytes, mock_site_check):
        """Regression: baseline_created_at was set after validation and lost its 'Z',
        so browsers showed it 5.5 h early in IST."""
        mock_site_check.extend([0.1, 0.2])
        url = f"/patients/{patient['id']}/site-check"
        first = client.post(url, files={"file": ("x.png", png_bytes, "image/png")}).json()
        second = client.post(url, files={"file": ("x.png", png_bytes, "image/png")}).json()
        assert second["baseline_checkin_id"] == first["id"]
        assert second["baseline_created_at"].endswith("Z")
        assert "since last check-in" in second["risk_velocity"]

    def test_repeated_upload_creates_a_separate_check_in(self, client, patient, png_bytes, mock_site_check):
        """Documented behaviour: re-uploading the same photo is a new check-in, not a dedupe."""
        mock_site_check.extend([0.2, 0.2])
        url = f"/patients/{patient['id']}/site-check"
        for _ in range(2):
            assert client.post(url, files={"file": ("x.png", png_bytes, "image/png")}).status_code == 201
        assert client.get(f"/patients/{patient['id']}/history").json()["n_checkins"] == 2

    def test_unknown_patient_site_check_404s(self, client, png_bytes, mock_site_check):
        response = client.post("/patients/99999/site-check", files={"file": ("x.png", png_bytes, "image/png")})
        assert response.status_code == 404
