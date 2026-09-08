"""Evaluate the trained vein segmenter on the held-out test split.

    python vein_module/src/evaluate.py

Produces docs/vein_module_eval_report.md with segmentation metrics, the
insertion-point distance against nurse-marked ground truth where CUBITAL
provides it, and example visualisations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from vein_module.src.augmentations import get_eval_transform
from vein_module.src.data_loader import VeinSegmentationDataset
from vein_module.src.model import AttentionUNet
from vein_module.src.site_selector import estimate_needle_angle, select_insertion_site
from vein_module.src.train import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

WITHIN_PX = 20.0  # the plan's success radius for insertion-point agreement


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precision, recall for one binary pair.

    An empty prediction on an empty target is a perfect score, not a division
    by zero - that case is a correct "no vein visible here".
    """
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)

    tp = float(np.logical_and(pred_b, target_b).sum())
    fp = float(np.logical_and(pred_b, ~target_b).sum())
    fn = float(np.logical_and(~pred_b, target_b).sum())

    if tp + fp + fn == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}

    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def load_nurse_annotations(path: Path | None) -> dict[str, tuple[float, float]]:
    """Load nurse-marked insertion points, keyed by image stem.

    CUBITAL ships 384 samples annotated by three certified nurses. Expected
    format is a JSON object mapping stem -> [x, y]. Returns an empty dict when
    absent, and the distance section of the report is skipped rather than
    fabricated.
    """
    if path is None or not path.exists():
        log.warning("no nurse annotations at %s - skipping distance metrics", path)
        return {}
    raw = json.loads(path.read_text())
    return {stem: (float(xy[0]), float(xy[1])) for stem, xy in raw.items()}


def overlay(image: np.ndarray, mask: np.ndarray, site: dict, truth=None) -> np.ndarray:
    """Original image + predicted mask + predicted site + nurse ground truth."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    tint = np.zeros_like(canvas)
    tint[mask.astype(bool)] = (0, 0, 255)          # red mask
    canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0)

    if site.get("x") is not None:
        cv2.drawMarker(canvas, (site["x"], site["y"]), (0, 255, 0),
                       cv2.MARKER_CROSS, 18, 2)     # green = predicted
    if truth is not None:
        cv2.drawMarker(canvas, (int(truth[0]), int(truth[1])), (255, 255, 0),
                       cv2.MARKER_TILTED_CROSS, 18, 2)  # cyan = nurse
    return canvas


def evaluate(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    figures_dir: Path,
    annotations_path: Path | None = None,
    threshold: float = 0.5,
    n_examples: int = 8,
) -> dict:
    device = get_device("auto")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    image_size = config.get("image_size", 512)

    model = AttentionUNet(base_channels=config.get("base_channels", 32)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = VeinSegmentationDataset(
        data_dir, "test", get_eval_transform(image_size), image_size
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    nurse_points = load_nurse_annotations(annotations_path)

    per_image, distances, confidences = [], [], []
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_examples = []

    with torch.inference_mode():
        for index, (image, mask) in enumerate(loader):
            probs = torch.sigmoid(model(image.to(device))).cpu().numpy()[0, 0]
            pred = (probs > threshold).astype(np.uint8)
            target = mask.numpy()[0, 0].astype(np.uint8)

            metrics = segmentation_metrics(pred, target)
            site = select_insertion_site(pred)
            confidences.append(site["confidence"])

            stem = dataset.pairs[index][0].stem
            truth = nurse_points.get(stem)
            if truth is not None and site["x"] is not None:
                distance = float(np.hypot(site["x"] - truth[0], site["y"] - truth[1]))
                distances.append(distance)
                metrics["distance_px"] = distance

            metrics["stem"] = stem
            metrics["site_found"] = site["x"] is not None
            metrics["target_empty"] = bool(target.sum() == 0)
            metrics["pred_empty"] = bool(pred.sum() == 0)
            per_image.append(metrics)

            if len(saved_examples) < n_examples:
                rgb = (image.numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                path = figures_dir / f"vein_example_{len(saved_examples):02d}.png"
                cv2.imwrite(str(path), overlay(bgr, pred, site, truth))
                saved_examples.append(path.name)

    def mean(key, subset=None):
        source = per_image if subset is None else subset
        values = [m[key] for m in source if key in m]
        return float(np.mean(values)) if values else 0.0

    # Split empty-target images out of the headline metric.
    #
    # An image with no annotated vein, where the model also predicts nothing,
    # scores a perfect Dice of 1.0. That is arguably correct behaviour, but
    # averaging it into the headline number does two bad things: it inflates the
    # score by the empty-mask fraction (4.6% of the CUBITAL test split), and it
    # *hides total collapse* - a model predicting background everywhere still
    # reports Dice 0.046 rather than 0.0, which looks like weak learning instead
    # of no learning. Both numbers are reported.
    with_vein = [m for m in per_image if not m["target_empty"]]
    without_vein = [m for m in per_image if m["target_empty"]]

    summary = {
        "n_test_images": len(per_image),
        "threshold": threshold,
        # Headline numbers: images that actually contain a vein.
        "n_with_vein": len(with_vein),
        "dice": mean("dice", with_vein),
        "iou": mean("iou", with_vein),
        "precision": mean("precision", with_vein),
        "recall": mean("recall", with_vein),
        # Same metrics over every image, for comparison with papers that do not
        # separate the two.
        "dice_all_images": mean("dice"),
        "iou_all_images": mean("iou"),
        # Empty-target images, scored as a correct-rejection rate instead.
        "n_without_vein": len(without_vein),
        "empty_correctly_rejected": (
            float(np.mean([m["pred_empty"] for m in without_vein]))
            if without_vein else 0.0
        ),
        "collapsed_to_background": float(np.mean([m["pred_empty"] for m in per_image])),
        "site_found_rate": float(np.mean([m["site_found"] for m in per_image])),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "checkpoint": str(checkpoint_path),
        "trained_epochs": checkpoint.get("epoch"),
        "examples": saved_examples,
    }
    if distances:
        summary.update(
            {
                "n_with_nurse_annotation": len(distances),
                "mean_distance_px": float(np.mean(distances)),
                "median_distance_px": float(np.median(distances)),
                "pct_within_20px": float(np.mean(np.array(distances) <= WITHIN_PX) * 100),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_report(summary, output_dir / "vein_module_eval_report.md")
    log.info("Dice %.4f  IoU %.4f  site found %.0f%%",
             summary["dice"], summary["iou"], summary["site_found_rate"] * 100)
    return summary


def _write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Vein Module — Evaluation Report",
        "",
        "Generated by `vein_module/src/evaluate.py`. Do not edit by hand.",
        "",
        f"Checkpoint: `{summary['checkpoint']}` (epoch {summary['trained_epochs']})",
        f"Test images: {summary['n_test_images']}  ·  threshold {summary['threshold']}",
        "",
        "## Segmentation",
        "",
        f"Scored over the **{summary['n_with_vein']} test images that contain an "
        f"annotated vein**. The other {summary['n_without_vein']} have empty "
        "masks and are reported separately below — averaging them into the "
        "headline would inflate it and could hide a model that predicts nothing.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Dice | {summary['dice']:.4f} |",
        f"| IoU | {summary['iou']:.4f} |",
        f"| Precision | {summary['precision']:.4f} |",
        f"| Recall | {summary['recall']:.4f} |",
        "",
        "| Diagnostic | Value |",
        "|---|---:|",
        f"| Dice incl. empty-mask images | {summary['dice_all_images']:.4f} |",
        f"| Empty masks correctly left blank | {summary['empty_correctly_rejected'] * 100:.0f}% |",
        f"| Images predicted entirely background | "
        f"{summary['collapsed_to_background'] * 100:.0f}% |",
        "",
        "A high value in that last row means the network collapsed to the "
        "majority class. Vein pixels are only ~0.45% of a CUBITAL frame, so "
        "predicting background everywhere is a strong local minimum — it is the "
        "first failure to check for, not a rounding detail.",
        "",
        "## Insertion site selection",
        "",
        f"A site was returned for **{summary['site_found_rate'] * 100:.0f}%** of test "
        f"images; mean confidence {summary['mean_confidence']:.3f}. Images with no "
        "site returned are not failures — the selector refuses when no segment "
        "meets the minimum width and straightness bar, which is the intended "
        "behaviour.",
        "",
    ]

    if "mean_distance_px" in summary:
        lines += [
            "### Agreement with nurse-marked points",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Annotated test images | {summary['n_with_nurse_annotation']} |",
            f"| Mean distance | {summary['mean_distance_px']:.1f} px |",
            f"| Median distance | {summary['median_distance_px']:.1f} px |",
            f"| Within {WITHIN_PX:.0f} px | {summary['pct_within_20px']:.1f}% |",
            "",
            "Nurse marks are a reference, not a gold standard: three nurses "
            "disagree with each other too, and no inter-rater spread is published "
            "alongside them. Read this as agreement with typical practice, not as "
            "accuracy against a true optimum.",
            "",
        ]
    else:
        lines += [
            "### Agreement with nurse-marked points",
            "",
            "> Not computed — no nurse annotation file was supplied. CUBITAL ships "
            "384 nurse-annotated samples; pass `--annotations` once they are "
            "available.",
            "",
        ]

    lines += ["## Examples", "",
              "Red = predicted vein mask · green cross = selected site · "
              "cyan cross = nurse-marked point.", ""]
    lines += [f"![example {i}](figures/{name})" for i, name in enumerate(summary["examples"])]
    lines += [
        "",
        "## Known failure: predictions land off the patient",
        "",
        "The overlays show vein-shaped false positives on the **bedsheet**, and "
        "on some images the selected insertion site is placed off the arm "
        "entirely. That is the direct cause of the precision/recall split above "
        "(recall high, precision low): the network over-predicts, and creased "
        "linen produces exactly the elongated ridge structure both the Frangi "
        "channel and the network are tuned to find.",
        "",
        "**A site marker on a bedsheet is a clinically unacceptable output**, "
        "not a cosmetic issue, and it is not fixed by thresholding.",
        "",
        "The fix is already available in the data and was discarded. CUBITAL "
        "masks are three-class — `0` background, `1` arm, `2` vein — and the "
        "loader keeps only class 2. Training the network to predict the arm as "
        "well, then intersecting the vein prediction with it, constrains "
        "candidates to the limb for free. That is the top priority for the next "
        "training run.",
        "",
        "## Limitations",
        "",
        "- **Under-trained.** This checkpoint is from epoch 6 of a planned 25. "
        "Validation Dice was still climbing monotonically when the run was "
        "stopped (0.133 → 0.179 → 0.204 → 0.217 → 0.240 → 0.353), so these "
        "numbers are a lower bound, not a converged result.",
        "- **Paediatric cohort.** CUBITAL is 1,008 subjects aged 5–22 (median "
        "11), 97% recorded as slim, 42 obese. The clinical need is in elderly "
        "and obese adults, where veins are harder to find. Transfer to that "
        "population is untested and should not be assumed.",
        "- Trained on NIR imagery. Deploying against an RGB camera is a domain "
        "shift this number does not measure.",
        "- The needle-angle output is a heuristic proxy for depth, not a "
        "measurement. See `site_selector.estimate_needle_angle`.",
        "- Test subjects are disjoint from training subjects, but all come from "
        "the same capture rig and population. Cross-dataset transfer is untested "
        "until U-DAVIS is available.",
        "",
    ]
    path.write_text("\n".join(lines))
    log.info("report -> %s", path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(repo_root / "vein_module/models/attention_unet_best.pt"),
    )
    parser.add_argument("--data-dir", default=str(repo_root / "data/processed/vein"))
    parser.add_argument("--output-dir", default=str(repo_root / "docs"))
    parser.add_argument("--figures-dir", default=str(repo_root / "docs/figures"))
    parser.add_argument(
        "--annotations",
        default=str(repo_root / "data/raw/cubital/nurse_annotations.json"),
        help="JSON mapping image stem -> [x, y] nurse-marked insertion point",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate(
        Path(args.checkpoint),
        Path(args.data_dir),
        Path(args.output_dir),
        Path(args.figures_dir),
        Path(args.annotations),
        args.threshold,
    )


if __name__ == "__main__":
    main()
