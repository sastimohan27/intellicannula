"""Gates that keep the system from scoring things it did not detect."""

from __future__ import annotations

import numpy as np
import pytest


class TestCannulaGate:
    def test_no_cannula_is_422_and_nothing_is_stored(self, client, patient, png_bytes, monkeypatch):
        """A photo with no hub must not get a risk score or become the baseline."""
        from app.api import infection
        from app.core.ml import NoCannulaDetected

        def fake(*_args):
            raise NoCannulaDetected("No cannula detected in this photo.")

        monkeypatch.setattr(infection, "run_site_check", fake)
        url = f"/patients/{patient['id']}/site-check"
        response = client.post(url, files={"file": ("x.png", png_bytes, "image/png")})

        assert response.status_code == 422
        assert "No cannula detected" in response.json()["detail"]
        assert client.get(f"/patients/{patient['id']}/history").json()["n_checkins"] == 0

    def test_detector_confidence_is_returned(self, client, patient, png_bytes, monkeypatch):
        from app.api import infection

        monkeypatch.setattr(infection, "run_site_check", lambda *_a: {
            "risk_score": 0.2, "roi_bbox": [1.0, 1.0, 9.0, 9.0], "color_distance": 3.0,
            "registration_degraded": False, "cannula_confidence": 0.78,
            "model_version": "lgbm-test",
        })
        response = client.post(
            f"/patients/{patient['id']}/site-check",
            files={"file": ("x.png", png_bytes, "image/png")},
        )
        assert response.json()["cannula_confidence"] == pytest.approx(0.78)

    def test_missing_detector_weights_is_503_not_a_centre_guess(self, monkeypatch, tmp_path):
        from app.core import ml

        monkeypatch.setattr(ml.settings, "repo_root", tmp_path)
        with pytest.raises(ml.ModelNotAvailable):
            ml._locate_roi(np.zeros((64, 64, 3), np.uint8))


class TestVeinMaskCleanup:
    def test_small_blobs_removed_large_vein_kept(self):
        from vein_module.src.postprocess import remove_small_components

        mask = np.zeros((64, 64), bool)
        mask[30:33, 5:60] = True        # 165 px vein-like run
        mask[5, 5] = mask[50:52, 50:52] = True  # 1 px and 4 px speckle
        cleaned = remove_small_components(mask, min_area=30)

        assert cleaned.dtype == np.uint8
        assert cleaned[30:33, 5:60].all()
        assert cleaned.sum() == 165

    def test_zero_area_is_a_no_op(self):
        from vein_module.src.postprocess import remove_small_components

        mask = np.random.default_rng(0).random((32, 32)) > 0.7
        assert (remove_small_components(mask, 0) == mask).all()


class TestVeinGrade:
    def test_grade_ignores_selector_separation(self):
        """Width decides the grade; a refusal is always Not Recommended."""
        from app.api.vein import _calculate_vein_grade

        assert _calculate_vein_grade(244, 13.8) == "Grade A (Excellent)"
        assert _calculate_vein_grade(244, 9.0) == "Grade B (Moderate)"
        assert _calculate_vein_grade(244, 5.0) == "Grade C (Difficult)"
        assert _calculate_vein_grade(None, None) == "Not Recommended"


class TestPhotoQuality:
    def _photo(self):
        rng = np.random.default_rng(0)
        return rng.integers(60, 200, (400, 400, 3), dtype=np.uint8)  # textured, mid-grey

    def test_sharp_well_lit_photo_passes(self):
        from app.core.ml import check_photo_quality

        assert check_photo_quality(self._photo())["sharpness"] > 20

    def test_blurred_photo_rejected(self):
        import cv2

        from app.core.ml import PhotoQualityError, check_photo_quality

        with pytest.raises(PhotoQualityError, match="blurred"):
            check_photo_quality(cv2.GaussianBlur(self._photo(), (0, 0), 6))

    def test_dark_photo_rejected(self):
        from app.core.ml import PhotoQualityError, check_photo_quality

        with pytest.raises(PhotoQualityError, match="too dark"):
            check_photo_quality((self._photo() * 0.15).astype(np.uint8))

    def test_overexposed_photo_rejected(self):
        from app.core.ml import PhotoQualityError, check_photo_quality

        img = self._photo()
        img[:250] = 255
        with pytest.raises(PhotoQualityError, match="overexposed"):
            check_photo_quality(img)
