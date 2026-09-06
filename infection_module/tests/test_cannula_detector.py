"""Tests for ROI cropping.

`detect_cannula` needs trained weights that require hand-labelled data, so it
is not exercised here. `crop_roi` is pure geometry and fully testable, and it
is the piece Phase 5 depends on for a stable ROI.
"""

import numpy as np
import pytest

from infection_module.src.cannula_detector.inference import crop_roi


class TestCropRoi:
    def test_margin_expands_the_box(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = crop_roi(image, [80, 80, 120, 120], margin_frac=0.5)
        assert crop.shape[0] == 80 and crop.shape[1] == 80  # 40 + 2*20

    def test_zero_margin_returns_the_bbox_exactly(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = crop_roi(image, [80, 80, 120, 120], margin_frac=0.0)
        assert crop.shape[:2] == (40, 40)

    def test_margin_is_proportional_not_fixed(self):
        """Same anatomical area whether shot close up or at arm's length."""
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        small = crop_roi(image, [100, 100, 120, 120], margin_frac=0.5)
        large = crop_roi(image, [100, 100, 180, 180], margin_frac=0.5)
        assert small.shape[0] * 4 == large.shape[0]

    def test_clips_at_the_frame_edge(self):
        """A hub near the edge yields an asymmetric crop, not an index error."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = crop_roi(image, [0, 0, 40, 40], margin_frac=0.5)
        assert crop.shape[0] == 60 and crop.shape[1] == 60

    def test_bbox_covering_the_whole_frame_is_clipped(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert crop_roi(image, [0, 0, 100, 100], margin_frac=0.5).shape[:2] == (100, 100)

    def test_degenerate_bbox_raises(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty crop"):
            crop_roi(image, [50, 50, 50, 50], margin_frac=0.0)

    def test_preserves_pixel_content(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[40:60, 40:60] = 255
        crop = crop_roi(image, [40, 40, 60, 60], margin_frac=0.0)
        assert (crop == 255).all()
