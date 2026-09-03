"""Tests for insertion site selection.

This module's output is what a clinician would act on, so the tests weight
failure behaviour at least as heavily as happy paths: an empty mask, a mask too
thin to be a credible target, and near-tied candidates must all be reported
rather than papered over with a confident-looking answer.
"""

import numpy as np
import pytest

from vein_module.src.site_selector import (
    MAX_ANGLE_DEG,
    MIN_ANGLE_DEG,
    estimate_needle_angle,
    select_insertion_site,
)


def _horizontal_vein(height=128, width=128, row=64, thickness=8, x0=10, x1=118):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[row - thickness // 2 : row + thickness // 2, x0:x1] = 255
    return mask


class TestSelectInsertionSiteHappyPath:
    def test_finds_a_point_on_a_straight_vein(self):
        result = select_insertion_site(_horizontal_vein())
        assert result["x"] is not None
        assert 55 < result["y"] < 73, "point should sit on the vein"

    def test_reports_plausible_width(self):
        result = select_insertion_site(_horizontal_vein(thickness=8))
        assert 6 <= result["vein_width_px"] <= 10

    def test_prefers_the_wider_of_two_veins(self):
        """The 0.5 width weight is the dominant term; this pins that behaviour."""
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[28:32, 10:118] = 255   # thin
        mask[60:74, 10:118] = 255   # thick
        result = select_insertion_site(mask)
        assert 60 <= result["y"] <= 74

    def test_vertical_vein_is_not_rejected(self):
        """A y-on-x line fit would score a vertical vein as unfittable."""
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[10:118, 60:68] = 255
        result = select_insertion_site(mask)
        assert result["x"] is not None, "vertical vein was wrongly rejected"


class TestSelectInsertionSiteFailureModes:
    def test_empty_mask_returns_no_site_with_a_reason(self):
        result = select_insertion_site(np.zeros((128, 128), dtype=np.uint8))
        assert result["x"] is None
        assert result["confidence"] == 0.0
        assert "empty mask" in result["reason"]

    def test_vein_below_minimum_width_is_refused(self):
        """A 2px vein is not a credible cannulation target."""
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[64:66, 10:118] = 255
        result = select_insertion_site(mask, min_width_px=4.0)
        assert result["x"] is None
        assert "minimum bar" in result["reason"]

    def test_segment_below_minimum_length_is_refused(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[60:70, 60:68] = 255  # a blob, not a run
        result = select_insertion_site(mask, min_length_px=30.0)
        assert result["x"] is None

    def test_single_candidate_never_claims_full_confidence(self):
        """One option is not evidence that the option is good."""
        result = select_insertion_site(_horizontal_vein())
        assert result["confidence"] <= 0.5
        assert "only one candidate" in result["reason"]

    def test_near_tied_candidates_yield_low_confidence(self):
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[40:48, 10:150] = 255
        mask[110:118, 10:150] = 255   # near-identical twin
        result = select_insertion_site(mask)
        assert result["confidence"] < 0.5
        assert result["n_candidates"] >= 2

    def test_all_white_mask_does_not_crash(self):
        result = select_insertion_site(np.full((64, 64), 255, dtype=np.uint8))
        assert "reason" in result

    def test_accepts_binary_as_well_as_0_255_masks(self):
        as_255 = select_insertion_site(_horizontal_vein())
        as_01 = select_insertion_site((_horizontal_vein() > 0).astype(np.uint8))
        assert as_255["x"] == as_01["x"] and as_255["y"] == as_01["y"]


class TestEstimateNeedleAngle:
    def test_wide_high_contrast_vein_gets_the_shallow_angle(self):
        assert estimate_needle_angle(20.0, 1.0) == pytest.approx(MIN_ANGLE_DEG)

    def test_narrow_low_contrast_vein_gets_the_steep_angle(self):
        assert estimate_needle_angle(3.0, 0.0) == pytest.approx(MAX_ANGLE_DEG)

    def test_always_within_clinical_bounds(self):
        for width in (-100.0, 0.0, 5.0, 50.0, 1e6):
            for contrast in (-5.0, 0.0, 0.5, 1.0, 99.0):
                angle = estimate_needle_angle(width, contrast)
                assert MIN_ANGLE_DEG <= angle <= MAX_ANGLE_DEG

    def test_monotonic_in_width(self):
        angles = [estimate_needle_angle(w, 0.5) for w in (3, 8, 13, 18, 20)]
        assert angles == sorted(angles, reverse=True)

    def test_monotonic_in_contrast(self):
        angles = [estimate_needle_angle(10.0, c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert angles == sorted(angles, reverse=True)
