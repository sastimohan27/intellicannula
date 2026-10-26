"""Train the Attention U-Net vein segmenter.

    python vein_module/src/train.py --epochs 100

Device selection prefers CUDA, then Apple Silicon MPS, then CPU. Mixed
precision is enabled on CUDA only - MPS autocast support is inconsistent and
silently falls back, so it is not worth the debugging surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vein_module.src.augmentations import get_eval_transform, get_train_transform
from vein_module.src.data_loader import VeinSegmentationDataset
from vein_module.src.model import AttentionUNet, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is a convenience, not a requirement

    class SummaryWriter:  # type: ignore[no-redef]
        """No-op stand-in so a missing tensorboard cannot abort a training run.

        The CSV log in the output directory carries the same per-epoch numbers,
        so nothing is lost except the live dashboard.
        """

        def __init__(self, *_args, **_kwargs):
            log.warning("tensorboard not installed - scalar logging goes to CSV only")

        def add_scalar(self, *_args, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DiceLoss(nn.Module):
    """Soft Dice on sigmoid probabilities.

    Paired with BCE rather than used alone. Veins cover a low single-digit
    percentage of a frame, so BCE alone is minimised well by predicting "no
    vein everywhere"; Dice is scale-invariant to that imbalance. Dice alone,
    though, has unstable gradients when a batch contains an empty mask, which
    is why the plan's 0.5/0.5 combination is the right default.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        numerator = 2.0 * (probs * targets).sum(dim=(1, 2, 3)) + self.smooth
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        return 1.0 - (numerator / denominator).mean()


def estimate_pos_weight(data_dir: Path, split: str = "train", cap: bool = True) -> float:
    """Positive-class weight for BCE, measured from the masks.

    Vein pixels are 0.358% of a CUBITAL frame - a 278:1 negative-to-positive
    ratio. Unweighted BCE is minimised almost perfectly by predicting background
    everywhere, and that is exactly what the network did: at epochs 1, 3 and 4 of
    the first run, val Dice and IoU were *identical* at 0.0559, which can only
    happen when true positives are zero. The non-zero value came entirely from
    empty masks scoring 1.0.

    Using the raw 278:1 ratio overcorrects and floods the prediction with false
    positives, so this returns sqrt(ratio) (~16.7 here) by default. That is a
    standard softening: strong enough to make the positive class matter, mild
    enough not to invert the problem. Pass cap=False for the raw ratio.
    """
    mask_dir = Path(data_dir) / split / "masks"
    positive = total = 0
    for path in sorted(mask_dir.glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        positive += int((mask > 127).sum())
        total += mask.size

    if positive == 0:
        log.warning("no positive pixels found in %s - using pos_weight 1.0", mask_dir)
        return 1.0

    fraction = positive / total
    ratio = (1.0 - fraction) / fraction
    weight = float(np.sqrt(ratio)) if cap else float(ratio)
    log.info(
        "positive pixels %.4f%% (neg/pos %.1f) -> pos_weight %.1f",
        fraction * 100, ratio, weight,
    )
    return weight


class CombinedLoss(nn.Module):
    """Weighted BCEWithLogits + Dice.

    Defaults are Dice-dominant (0.3 / 0.7). Dice is scale-invariant to class
    imbalance, so it keeps pulling toward overlap while BCE - even weighted -
    still rewards the all-background solution. The first training run used an
    even 0.5/0.5 split with no pos_weight and collapsed; both changes address
    the same 278:1 imbalance from different directions.
    """

    def __init__(self, bce_weight: float = 0.3, pos_weight: float | None = None):
        super().__init__()
        weight_tensor = (
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None
        )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (
            1.0 - self.bce_weight
        ) * self.dice(logits, targets)


@torch.no_grad()
def dice_and_iou(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float]:
    """Hard Dice and IoU at a fixed threshold, averaged over the batch."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
    union = pred_sum + target_sum - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return float(dice.mean()), float(iou.mean())


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    image_size: int = 512
    base_channels: int = 32
    bce_weight: float = 0.3
    pos_weight: float | None = None      # None -> measured from the training masks
    patience: int = 15
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over `loader`. Training when `optimizer` is given, else eval."""
    training = optimizer is not None
    model.train(training)

    totals = np.zeros(3)  # loss, dice, iou
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            if scaler is not None and training:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, masks)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        dice, iou = dice_and_iou(logits.float(), masks)
        totals += np.array([loss.item(), dice, iou])
        n_batches += 1

    return tuple(totals / max(n_batches, 1))


def train(
    data_dir: Path, output_dir: Path, config: TrainConfig, run_dir: Path,
    resume: Path | None = None,
) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = get_device(config.device)
    log.info("device: %s", device)

    train_ds = VeinSegmentationDataset(
        data_dir, "train", get_train_transform(config.image_size), config.image_size
    )
    val_ds = VeinSegmentationDataset(
        data_dir, "val", get_eval_transform(config.image_size), config.image_size
    )

    # persistent_workers only helps when workers exist; pin_memory is a CUDA
    # concept and warns on MPS.
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    model = AttentionUNet(base_channels=config.base_channels).to(device)
    log.info("parameters: %s", f"{count_parameters(model):,}")

    pos_weight = (
        config.pos_weight
        if config.pos_weight is not None
        else estimate_pos_weight(data_dir)
    )
    criterion = CombinedLoss(config.bce_weight, pos_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    csv_path = output_dir / "training_log.csv"
    checkpoint_path = output_dir / "attention_unet_best.pt"

    best_dice, best_epoch, epochs_without_improvement = -1.0, -1, 0
    start_epoch = 1

    if resume is not None:
        # Continue an interrupted run: weights, best-so-far, and the cosine LR
        # position. Optimizer moments are restored when the checkpoint has them
        # (older checkpoints did not save them; AdamW re-warms within an epoch).
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        best_dice, best_epoch = state.get("val_dice", -1.0), state["epoch"]
        for _ in range(state["epoch"]):
            scheduler.step()
        log.info("resumed from %s at epoch %d (val Dice %.4f)", resume, state["epoch"], best_dice)
    else:
        with csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["epoch", "train_loss", "train_dice", "train_iou",
                 "val_loss", "val_dice", "val_iou", "lr", "seconds"]
            )

    epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):
        started = time.time()
        train_loss, train_dice, train_iou = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - started

        for name, value in [
            ("loss/train", train_loss), ("loss/val", val_loss),
            ("dice/train", train_dice), ("dice/val", val_dice),
            ("iou/train", train_iou), ("iou/val", val_iou),
        ]:
            writer.add_scalar(name, value, epoch)

        with csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                epoch, f"{train_loss:.5f}", f"{train_dice:.5f}", f"{train_iou:.5f}",
                f"{val_loss:.5f}", f"{val_dice:.5f}", f"{val_iou:.5f}",
                f"{scheduler.get_last_lr()[0]:.2e}", f"{elapsed:.1f}",
            ])

        log.info(
            "epoch %3d/%d  train loss %.4f dice %.4f | val loss %.4f dice %.4f iou %.4f  (%.1fs)",
            epoch, config.epochs, train_loss, train_dice, val_loss, val_dice, val_iou, elapsed,
        )

        if val_dice > best_dice:
            best_dice, best_epoch, epochs_without_improvement = val_dice, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
            log.info("  new best val Dice %.4f -> %s", val_dice, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                log.info(
                    "early stopping: no val Dice improvement in %d epochs (best %.4f @ epoch %d)",
                    config.patience, best_dice, best_epoch,
                )
                break

    writer.close()

    summary = {
        **asdict(config),
        "pos_weight_used": pos_weight,
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "parameters": count_parameters(model),
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "training_config.json").write_text(json.dumps(summary, indent=2))
    log.info("best val Dice %.4f at epoch %d", best_dice, best_epoch)
    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(repo_root / "data/processed/vein"))
    parser.add_argument("--output-dir", default=str(repo_root / "vein_module/models"))
    parser.add_argument("--run-dir", default=str(repo_root / "runs/vein_unet"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument(
        "--pos-weight", type=float, default=None,
        help="BCE positive-class weight; omit to measure it from the masks",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--resume", default=None, help="checkpoint to continue training from")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        patience=args.patience,
        bce_weight=args.bce_weight,
        pos_weight=args.pos_weight,
        num_workers=args.num_workers,
        device=args.device,
    )
    train(
        Path(args.data_dir), Path(args.output_dir), config, Path(args.run_dir),
        resume=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
