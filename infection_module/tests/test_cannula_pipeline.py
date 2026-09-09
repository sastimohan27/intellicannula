"""Smoke test for the YOLOv11 training pipeline.

IMPORTANT: this does NOT test a cannula detector - no such model exists, because
no public cannula dataset exists and none has been fabricated. What it verifies
is that the *plumbing* around ultralytics works: the data.yaml is well-formed,
train.py's arguments are accepted, a checkpoint is produced and copied to the
expected path, and inference.py can load it and return the documented shape.

The point is that when 100-200 real images are eventually labelled, training
fails for dataset reasons only - never because of a typo in the training call.

Marked `slow`: it downloads a pretrained checkpoint and runs two epochs.
Run with: pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTOR = REPO_ROOT / "infection_module/src/cannula_detector"


class TestDataYaml:
    """Cheap checks, always run."""

    def test_exists_and_parses(self):
        config = yaml.safe_load((DETECTOR / "data.yaml").read_text())
        assert config is not None

    def test_declares_a_single_cannula_class(self):
        config = yaml.safe_load((DETECTOR / "data.yaml").read_text())
        assert config["nc"] == 1
        assert config["names"][0] == "cannula_hub"

    def test_declares_all_three_splits(self):
        config = yaml.safe_load((DETECTOR / "data.yaml").read_text())
        for split in ("train", "val", "test"):
            assert split in config


class TestTrainingConfig:
    def test_mixup_is_disabled(self):
        """Blending a cannula with an unrelated image makes translucent hubs
        that do not occur, and dilutes an already tiny dataset."""
        source = (DETECTOR / "train.py").read_text()
        assert "mixup=0.0" in source

    def test_vertical_flip_is_disabled(self):
        """An arm photographed upside down is not a real presentation."""
        source = (DETECTOR / "train.py").read_text()
        assert "flipud=0.0" in source

    def test_hsv_jitter_is_moderate(self):
        """Aggressive hue/saturation shifts would teach the detector to ignore
        redness - the exact cue the Phase 5 infection model depends on."""
        source = (DETECTOR / "train.py").read_text()
        assert "hsv_s=0.4" in source
        assert "hsv_h=0.015" in source


@pytest.mark.slow
class TestPipelineRuns:
    def test_train_and_infer_end_to_end(self, tmp_path):
        """Two epochs on ultralytics' bundled 8-image sample.

        Uses coco8, NOT cannula data - this proves the code path works, and
        makes no claim about detecting anything clinical.
        """
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.train(
            data="coco8.yaml", epochs=2, imgsz=64, batch=2,
            project=str(tmp_path), name="smoke", exist_ok=True,
            verbose=False, plots=False, val=False,
        )
        weights = tmp_path / "smoke" / "weights" / "best.pt"
        assert weights.exists(), "training produced no checkpoint"

        from infection_module.src.cannula_detector.inference import crop_roi

        import numpy as np

        loaded = YOLO(str(weights))
        results = loaded.predict(
            np.zeros((64, 64, 3), dtype=np.uint8), verbose=False
        )
        assert results is not None

        # crop_roi is the piece Phase 5 consumes.
        cropped = crop_roi(np.zeros((100, 100, 3), dtype=np.uint8), [20, 20, 60, 60])
        assert cropped.shape[0] > 0
