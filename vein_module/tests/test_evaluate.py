"""Tests for evaluation.

Includes a full train -> evaluate round trip on a tiny fixture, which is the
only way to catch a checkpoint/model shape mismatch before real training burns
hours.
"""

import json

import cv2
import numpy as np
import pytest

from vein_module.src.data_loader import prepare_vein_dataset
from vein_module.src.evaluate import (
    evaluate,
    load_nurse_annotations,
    segmentation_metrics,
)
from vein_module.src.train import TrainConfig, train


class TestSegmentationMetrics:
    def test_perfect_overlap(self):
        target = np.zeros((16, 16), dtype=np.uint8)
        target[4:12, 4:12] = 1
        m = segmentation_metrics(target, target)
        assert m["dice"] == 1.0 and m["iou"] == 1.0
        assert m["precision"] == 1.0 and m["recall"] == 1.0

    def test_disjoint_prediction_scores_zero(self):
        target = np.zeros((16, 16), dtype=np.uint8)
        target[0:4, 0:4] = 1
        pred = np.zeros((16, 16), dtype=np.uint8)
        pred[12:16, 12:16] = 1
        m = segmentation_metrics(pred, target)
        assert m["dice"] == 0.0 and m["iou"] == 0.0

    def test_both_empty_is_a_perfect_score_not_a_zero_division(self):
        """A correct 'no vein visible' must not be punished."""
        empty = np.zeros((16, 16), dtype=np.uint8)
        m = segmentation_metrics(empty, empty)
        assert m["dice"] == 1.0 and m["precision"] == 1.0

    def test_over_prediction_lowers_precision_not_recall(self):
        target = np.zeros((16, 16), dtype=np.uint8)
        target[4:8, 4:8] = 1
        m = segmentation_metrics(np.ones((16, 16), dtype=np.uint8), target)
        assert m["recall"] == 1.0
        assert m["precision"] < 0.2

    def test_under_prediction_lowers_recall_not_precision(self):
        target = np.zeros((16, 16), dtype=np.uint8)
        target[4:12, 4:12] = 1
        pred = np.zeros((16, 16), dtype=np.uint8)
        pred[4:6, 4:6] = 1
        m = segmentation_metrics(pred, target)
        assert m["precision"] == 1.0
        assert m["recall"] < 0.2

    def test_iou_never_exceeds_dice(self):
        rng = np.random.default_rng(0)
        pred = (rng.random((32, 32)) > 0.5).astype(np.uint8)
        target = (rng.random((32, 32)) > 0.5).astype(np.uint8)
        m = segmentation_metrics(pred, target)
        assert m["iou"] <= m["dice"] + 1e-9


class TestNurseAnnotations:
    def test_missing_file_returns_empty_rather_than_raising(self, tmp_path):
        """A missing annotation file must skip the metric, never fabricate it."""
        assert load_nurse_annotations(tmp_path / "absent.json") == {}
        assert load_nurse_annotations(None) == {}

    def test_parses_stem_to_xy(self, tmp_path):
        path = tmp_path / "nurse.json"
        path.write_text(json.dumps({"cubital_0001_left": [120, 240]}))
        assert load_nurse_annotations(path) == {"cubital_0001_left": (120.0, 240.0)}


@pytest.fixture
def trained(tmp_path):
    """Train a 2-epoch model on a tiny fixture so evaluate has a real checkpoint."""
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "masks").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for subject in range(12):
        for arm in ("left", "right"):
            stem = f"{subject:04d}_{arm}"
            img = rng.integers(80, 200, (64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[28:36, 8:56] = 2
            cv2.imwrite(str(raw / "images" / f"{stem}.png"), img)
            cv2.imwrite(str(raw / "masks" / f"{stem}.png"), mask)

    processed = tmp_path / "processed"
    prepare_vein_dataset(raw, tmp_path / "none", processed, image_size=64)

    config = TrainConfig(
        epochs=2, batch_size=2, image_size=64, base_channels=4,
        num_workers=0, device="cpu",
    )
    train(processed, tmp_path / "models", config, tmp_path / "runs")
    return processed, tmp_path / "models" / "attention_unet_best.pt", tmp_path


class TestEvaluateRoundTrip:
    def test_produces_a_report_and_metrics(self, trained):
        processed, checkpoint, tmp_path = trained
        summary = evaluate(
            checkpoint, processed, tmp_path / "docs", tmp_path / "docs" / "figures"
        )
        assert summary["n_test_images"] > 0
        assert 0.0 <= summary["dice"] <= 1.0
        assert (tmp_path / "docs" / "vein_module_eval_report.md").exists()

    def test_saves_example_overlays(self, trained):
        processed, checkpoint, tmp_path = trained
        summary = evaluate(
            checkpoint, processed, tmp_path / "docs", tmp_path / "docs" / "figures"
        )
        assert summary["examples"]
        for name in summary["examples"]:
            assert (tmp_path / "docs" / "figures" / name).exists()

    def test_report_flags_absent_nurse_annotations(self, trained):
        """Missing ground truth must be stated, not silently omitted."""
        processed, checkpoint, tmp_path = trained
        evaluate(checkpoint, processed, tmp_path / "docs", tmp_path / "docs" / "figures")
        report = (tmp_path / "docs" / "vein_module_eval_report.md").read_text()
        assert "Not computed" in report
        assert "Limitations" in report
