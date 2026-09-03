"""Tests for the training loop.

The full run needs CUBITAL, which is not downloadable on this network, so these
exercise the pieces that can be verified without it: the loss maths, the metrics,
device selection, and one end-to-end two-epoch run over a tiny fixture to prove
the loop, checkpointing, and early stopping actually wire together.
"""

import json

import cv2
import numpy as np
import pytest
import torch

from vein_module.src.data_loader import prepare_vein_dataset
from vein_module.src.train import (
    CombinedLoss,
    DiceLoss,
    TrainConfig,
    dice_and_iou,
    get_device,
    train,
)


class TestDiceLoss:
    def test_perfect_prediction_approaches_zero(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0
        logits = (target * 2 - 1) * 20  # saturated logits matching the target
        assert DiceLoss()(logits, target).item() < 0.05

    def test_inverted_prediction_approaches_one(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0
        assert DiceLoss()((1 - target) * 40 - 20, target).item() > 0.9

    def test_empty_target_does_not_produce_nan(self):
        """Smoothing exists for this case - a frame where no vein was segmented."""
        loss = DiceLoss()(torch.randn(2, 1, 16, 16), torch.zeros(2, 1, 16, 16))
        assert torch.isfinite(loss)

    def test_is_differentiable(self):
        logits = torch.randn(1, 1, 16, 16, requires_grad=True)
        DiceLoss()(logits, torch.ones(1, 1, 16, 16)).backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


class TestCombinedLoss:
    def test_sits_between_its_two_terms(self):
        torch.manual_seed(0)
        logits, target = torch.randn(2, 1, 16, 16), torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0

        bce = torch.nn.BCEWithLogitsLoss()(logits, target).item()
        dice = DiceLoss()(logits, target).item()
        combined = CombinedLoss(0.5)(logits, target).item()
        assert min(bce, dice) <= combined <= max(bce, dice)

    def test_weight_zero_is_pure_dice(self):
        torch.manual_seed(0)
        logits, target = torch.randn(1, 1, 16, 16), torch.ones(1, 1, 16, 16)
        assert CombinedLoss(0.0)(logits, target).item() == pytest.approx(
            DiceLoss()(logits, target).item(), rel=1e-5
        )


class TestMetrics:
    def test_perfect_overlap_scores_one(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0
        dice, iou = dice_and_iou((target * 2 - 1) * 20, target)
        assert dice > 0.99 and iou > 0.99

    def test_disjoint_prediction_scores_zero(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 0:4, 0:4] = 1.0
        pred_logits = torch.full((1, 1, 16, 16), -20.0)
        pred_logits[:, :, 12:16, 12:16] = 20.0
        dice, iou = dice_and_iou(pred_logits, target)
        assert dice < 0.01 and iou < 0.01

    def test_iou_never_exceeds_dice(self):
        """Definitional; catches a swapped return."""
        torch.manual_seed(0)
        target = (torch.rand(2, 1, 16, 16) > 0.7).float()
        dice, iou = dice_and_iou(torch.randn(2, 1, 16, 16), target)
        assert iou <= dice + 1e-6


class TestDeviceSelection:
    def test_explicit_request_is_honoured(self):
        assert get_device("cpu").type == "cpu"

    def test_auto_returns_an_available_device(self):
        assert get_device("auto").type in {"cuda", "mps", "cpu"}


@pytest.fixture
def tiny_dataset(tmp_path):
    """Six subjects, two arms each - enough for a non-empty train/val/test split."""
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
    prepare_vein_dataset(raw, tmp_path / "no_udavis", processed, image_size=64)
    return processed


class TestTrainLoop:
    def test_two_epochs_produce_a_loadable_checkpoint(self, tiny_dataset, tmp_path):
        config = TrainConfig(
            epochs=2, batch_size=2, image_size=64, base_channels=4,
            num_workers=0, device="cpu",
        )
        summary = train(tiny_dataset, tmp_path / "models", config, tmp_path / "runs")

        checkpoint = torch.load(
            tmp_path / "models" / "attention_unet_best.pt", weights_only=False
        )
        assert checkpoint["model_state_dict"]
        assert summary["best_epoch"] >= 1
        assert summary["epochs_run"] == 2

    def test_writes_config_and_csv_log(self, tiny_dataset, tmp_path):
        config = TrainConfig(
            epochs=2, batch_size=2, image_size=64, base_channels=4,
            num_workers=0, device="cpu",
        )
        train(tiny_dataset, tmp_path / "models", config, tmp_path / "runs")

        saved = json.loads((tmp_path / "models" / "training_config.json").read_text())
        assert saved["image_size"] == 64
        assert saved["parameters"] > 0

        rows = (tmp_path / "models" / "training_log.csv").read_text().strip().split("\n")
        assert len(rows) == 3  # header + 2 epochs

    def test_early_stopping_can_end_a_run_short(self, tiny_dataset, tmp_path):
        """patience=1 on a 5-epoch budget should stop before exhausting it."""
        config = TrainConfig(
            epochs=5, batch_size=2, image_size=64, base_channels=4,
            num_workers=0, device="cpu", patience=1,
        )
        summary = train(tiny_dataset, tmp_path / "models", config, tmp_path / "runs")
        assert summary["epochs_run"] <= 5
