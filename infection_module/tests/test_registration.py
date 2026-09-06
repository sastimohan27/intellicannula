"""Tests for baseline registration."""

import cv2
import numpy as np

from infection_module.src.registration import register_images


def _textured(seed=0, shape=(200, 200)):
    """Feature-rich image - ORB needs corners to match on."""
    rng = np.random.default_rng(seed)
    img = np.full((*shape, 3), 160, dtype=np.uint8)
    for _ in range(60):
        x, y = rng.integers(10, shape[1] - 20), rng.integers(10, shape[0] - 20)
        cv2.rectangle(img, (x, y), (x + 10, y + 10),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


class TestRegistration:
    def test_output_matches_baseline_dimensions(self):
        baseline = _textured()
        current = _textured(seed=1, shape=(150, 250))
        assert register_images(baseline, current).shape[:2] == baseline.shape[:2]

    def test_recovers_a_known_translation(self):
        baseline = _textured()
        shifted = cv2.warpAffine(
            baseline, np.float32([[1, 0, 12], [0, 1, 8]]), (200, 200)
        )
        warped, info = register_images(baseline, shifted, return_info=True)
        assert info["method"] == "homography"
        # Compare the interior, away from the border the warp leaves blank.
        assert float(np.abs(
            warped[30:170, 30:170].astype(float) - baseline[30:170, 30:170].astype(float)
        ).mean()) < 12

    def test_featureless_pair_falls_back_and_says_so(self):
        """A smooth forearm is the common case; a wrong homography is worse
        than an honest translation."""
        flat = np.full((200, 200, 3), 150, dtype=np.uint8)
        _, info = register_images(flat, flat.copy(), return_info=True)
        assert info["method"] == "translation_fallback"
        assert info["degraded"] is True

    def test_fallback_uses_the_bbox_anchor(self):
        flat = np.full((200, 200, 3), 150, dtype=np.uint8)
        warped = register_images(flat, flat.copy(), roi_bbox=[40, 40, 80, 80])
        assert warped.shape == flat.shape

    def test_zero_match_case_does_not_raise(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        warped, info = register_images(black, white, return_info=True)
        assert warped.shape == black.shape
        assert info["degraded"] is True

    def test_grayscale_input_accepted(self):
        baseline = cv2.cvtColor(_textured(), cv2.COLOR_BGR2GRAY)
        assert register_images(baseline, baseline.copy()).shape == baseline.shape
