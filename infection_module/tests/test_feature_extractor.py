"""Tests for colour and texture features.

Weighted toward the invariance properties the module exists to provide: the
same inflammation on different skin tones must produce a similar erythema
distance, and edge cases (black ROI, ROI filling the frame) must fail loudly
rather than return a plausible-looking number.
"""

import numpy as np
import pytest

from infection_module.src.feature_extractor import (
    color_features,
    extract_all_features,
    feature_names,
    get_baseline_patch,
    texture_features,
)


def _skin(shape=(60, 60), bgr=(150, 170, 200)):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


class TestGetBaselinePatch:
    def test_samples_outside_the_roi(self):
        image = _skin((200, 200))
        patch = get_baseline_patch(image, [80, 80, 120, 120])
        assert patch.size > 0

    def test_falls_back_to_another_side_near_the_frame_edge(self):
        """A site photographed at the top of the frame still needs a reference."""
        image = _skin((200, 200))
        patch = get_baseline_patch(image, [80, 2, 120, 42])
        assert patch.size > 0

    def test_raises_when_roi_fills_the_frame(self):
        """Fabricating a reference would corrupt every colour distance."""
        image = _skin((60, 60))
        with pytest.raises(ValueError, match="Retake"):
            get_baseline_patch(image, [0, 0, 60, 60])


class TestColorFeatures:
    def test_identical_roi_and_patch_give_near_zero_distance(self):
        skin = _skin()
        f = color_features(skin, skin)
        assert f["color_lab_distance"] < 1e-6
        assert abs(f["color_a_delta"]) < 1e-6

    def test_redder_roi_increases_distance_and_a_delta(self):
        healthy = _skin(bgr=(150, 170, 200))
        inflamed = _skin(bgr=(120, 130, 235))  # more red in BGR
        f = color_features(inflamed, healthy)
        assert f["color_lab_distance"] > 5
        assert f["color_a_delta"] > 0, "a* should rise toward red"

    def test_erythema_is_comparable_across_skin_tones(self):
        """The whole point of a per-image reference.

        The same inflammation on light and dark skin must yield a similar
        erythema distance; an absolute redness measure would not.
        """
        light_healthy = _skin(bgr=(170, 190, 215))
        light_inflamed = _skin(bgr=(150, 165, 235))
        dark_healthy = _skin(bgr=(60, 75, 105))
        dark_inflamed = _skin(bgr=(45, 55, 125))

        light = color_features(light_inflamed, light_healthy)["color_a_delta"]
        dark = color_features(dark_inflamed, dark_healthy)["color_a_delta"]
        assert light > 0 and dark > 0
        assert abs(light - dark) < max(light, dark) * 0.75

    def test_black_roi_does_not_crash(self):
        f = color_features(np.zeros((30, 30, 3), dtype=np.uint8), _skin())
        assert np.isfinite(f["color_lab_distance"])

    def test_all_values_finite(self):
        f = color_features(_skin(), np.zeros((10, 10, 3), dtype=np.uint8))
        assert all(np.isfinite(v) for v in f.values())


class TestTextureFeatures:
    def test_lbp_histogram_sums_to_one(self):
        f = texture_features(np.random.default_rng(0).integers(0, 255, (40, 40), dtype=np.uint8))
        lbp = [v for k, v in f.items() if k.startswith("texture_lbp_")]
        assert len(lbp) == 10  # uniform LBP with P=8 gives P+2 bins
        assert sum(lbp) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_region_has_lower_contrast_than_noise(self):
        flat = texture_features(np.full((40, 40), 128, dtype=np.uint8))
        noisy = texture_features(
            np.random.default_rng(0).integers(0, 255, (40, 40), dtype=np.uint8)
        )
        assert flat["texture_glcm_contrast_d1"] < noisy["texture_glcm_contrast_d1"]

    def test_all_black_roi_produces_finite_features(self):
        f = texture_features(np.zeros((32, 32), dtype=np.uint8))
        assert all(np.isfinite(v) for v in f.values())

    def test_accepts_colour_input(self):
        assert texture_features(_skin((40, 40)))

    def test_rotation_invariance_of_glcm(self):
        """GLCM is averaged over four angles, so a rotated ROI scores similarly."""
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (40, 40), dtype=np.uint8)
        a = texture_features(base)["texture_glcm_contrast_d1"]
        b = texture_features(np.rot90(base).copy())["texture_glcm_contrast_d1"]
        assert abs(a - b) < 0.25 * max(a, b)


class TestExtractAllFeatures:
    def test_keys_are_prefixed_and_stable(self):
        image = _skin((200, 200))
        f = extract_all_features(image, [80, 80, 120, 120])
        assert sorted(f.keys()) == feature_names()
        assert all(k.startswith(("color_", "texture_")) for k in f)

    def test_samples_its_own_baseline_patch(self):
        image = _skin((200, 200))
        assert extract_all_features(image, [80, 80, 120, 120])["color_lab_distance"] >= 0

    def test_empty_roi_raises(self):
        with pytest.raises(ValueError, match="empty"):
            extract_all_features(_skin((100, 100)), [50, 50, 50, 50])

    def test_feature_count_is_stable(self):
        assert len(feature_names()) == 30
