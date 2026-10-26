"""Error analysis of the deployed vein configuration on the TEST split.

    infection_module/.venv/bin/python -m vein_module.src.error_analysis

Analysis only: the configuration (epoch-16 checkpoint, threshold 0.9, 60 px
speckle filter) was fixed on the validation split before this runs, and
nothing here feeds back into it. Writes docs/vein_error_analysis.md and
docs/figures/vein_err_*.png.

Uses CUBITAL's three-class raw masks (0 background, 1 arm, 2 vein) to say
*where* errors fall, which the binary training masks cannot.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vein_module.src.augmentations import get_eval_transform
from vein_module.src.evaluate import segmentation_metrics
from vein_module.src.model import AttentionUNet
from vein_module.src.postprocess import binarize
from vein_module.src.train import get_device

ROOT = Path(__file__).resolve().parents[2]
RAW_MASKS = ROOT / "data/raw/cubital/square_dataset/masks"
THRESHOLD, MIN_AREA = 0.9, 60
NEAR_PX = 2        # an FP within this many px of a true vein is a margin error
THIN_PX = 4.0      # true-vein local width below this counts as a thin vein


def _leakage_report() -> dict:
    """Subject and pixel-identical overlap between splits."""
    manifest = json.loads((ROOT / "data/processed/vein/split_manifest.json").read_text())
    subjects: dict[str, set] = {}
    hashes: dict[str, set] = {}
    for split in ("train", "val", "test"):
        images = sorted((ROOT / f"data/processed/vein/{split}/images").glob("*.png"))
        hashes[split] = {hashlib.sha256(p.read_bytes()).hexdigest() for p in images}
    for split, members in manifest.get("subjects", {}).items():
        subjects[split] = set(members)
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        "identical_images_across_splits": {f"{a}/{b}": len(hashes[a] & hashes[b]) for a, b in pairs},
        "subjects_in_multiple_splits": (
            {f"{a}/{b}": len(subjects.get(a, set()) & subjects.get(b, set())) for a, b in pairs}
            if subjects else "subject assignment not found in split_manifest.json"
        ),
    }


def main() -> None:
    device = get_device("auto")
    ckpt = torch.load(ROOT / "vein_module/models/attention_unet_best.pt", map_location=device, weights_only=False)
    model = AttentionUNet(base_channels=ckpt["config"].get("base_channels", 32)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = get_eval_transform(512)

    rows, panels, latencies = [], {}, []
    tot = dict(tp=0, fp_near=0, fp_far_arm=0, fp_off_arm=0, fn_thin=0, fn_wide=0, gt_thin=0, gt_wide=0)
    misaligned = []
    for path in sorted((ROOT / "data/processed/vein/test/images").glob("cubital_*.png")):
        frame = path.stem.removeprefix("cubital_")
        raw = cv2.imread(str(RAW_MASKS / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        target = cv2.resize(cv2.imread(str(path).replace("/images/", "/masks/"), 0), (512, 512),
                            interpolation=cv2.INTER_NEAREST) > 127
        if raw is None or target.sum() == 0:
            continue
        raw = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        arm = raw >= 1

        bgr = cv2.imread(str(path))
        x = transform(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["image"]
        t = torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        with torch.inference_mode():
            start = time.perf_counter()
            probs = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            latencies.append(time.perf_counter() - start)
        pred = binarize(probs, THRESHOLD, MIN_AREA).astype(bool)

        # Pairing sanity: annotated veins must lie on the annotated arm.
        if (target & ~arm).sum() > 0.05 * target.sum():
            misaligned.append(frame)

        near = cv2.dilate(target.astype(np.uint8), np.ones((2 * NEAR_PX + 1,) * 2, np.uint8)).astype(bool)
        fp = pred & ~target
        width = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 3) * 2
        # Local width of each vein pixel = width at the nearest centreline; the
        # max over a 5x5 neighbourhood approximates it for boundary pixels.
        width = cv2.dilate(width, np.ones((5, 5), np.uint8)) * target
        thin = target & (width < THIN_PX)
        tot["tp"] += int((pred & target).sum())
        tot["fp_near"] += int((fp & near).sum())
        tot["fp_far_arm"] += int((fp & ~near & arm).sum())
        tot["fp_off_arm"] += int((fp & ~arm).sum())
        tot["fn_thin"] += int((thin & ~pred).sum())
        tot["fn_wide"] += int((target & ~thin & ~pred).sum())
        tot["gt_thin"] += int(thin.sum())
        tot["gt_wide"] += int((target & ~thin).sum())

        metrics = segmentation_metrics(pred, target)
        lab_l = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
        metrics.update({
            "frame": frame,
            "contrast": float(lab_l[arm].std()) if arm.any() else 0.0,
            "pred_components": cv2.connectedComponents(pred.astype(np.uint8))[0] - 1,
            "gt_components": cv2.connectedComponents(target.astype(np.uint8))[0] - 1,
        })
        rows.append(metrics)
        panels[frame] = (bgr, target, pred)

    dice = np.array([r["dice"] for r in rows])
    order = np.argsort(dice)
    picks = {"worst": order[0], "low": order[len(order) // 10], "median": order[len(order) // 2],
             "best": order[-1]}
    figs = ROOT / "docs/figures"
    figs.mkdir(parents=True, exist_ok=True)
    for tag, index in picks.items():
        frame = rows[index]["frame"]
        bgr, target, pred = panels[frame]
        overlay = bgr.copy()
        overlay[target & pred] = (0, 255, 255)    # yellow: correct
        overlay[pred & ~target] = (0, 0, 255)     # red: false positive
        overlay[target & ~pred] = (255, 128, 0)   # blue: missed
        gt_img = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        pr_img = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        strip = np.hstack([bgr, gt_img, pr_img, overlay])
        cv2.imwrite(str(figs / f"vein_err_{tag}.png"), cv2.resize(strip, (1600, 400)))

    terciles = np.quantile([r["contrast"] for r in rows], [1 / 3, 2 / 3])
    by_contrast = {}
    for name, lo, hi in [("low", -1, terciles[0]), ("mid", terciles[0], terciles[1]), ("high", terciles[1], 1e9)]:
        sel = [r["dice"] for r in rows if lo < r["contrast"] <= hi]
        by_contrast[name] = (float(np.mean(sel)), len(sel))

    fp_total = tot["fp_near"] + tot["fp_far_arm"] + tot["fp_off_arm"]
    frag = [r["pred_components"] / max(r["gt_components"], 1) for r in rows]
    summary = {
        "n": len(rows), "dice_mean": float(dice.mean()), "dice_median": float(np.median(dice)),
        "dice_p10": float(np.percentile(dice, 10)), "dice_p90": float(np.percentile(dice, 90)),
        "share_dice_below_0.4": float((dice < 0.4).mean()),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
        "fp_share": {k: tot[k] / fp_total for k in ("fp_near", "fp_far_arm", "fp_off_arm")},
        "recall_thin": 1 - tot["fn_thin"] / max(tot["gt_thin"], 1),
        "recall_wide": 1 - tot["fn_wide"] / max(tot["gt_wide"], 1),
        "thin_share_of_vein_px": tot["gt_thin"] / (tot["gt_thin"] + tot["gt_wide"]),
        "fragmentation_median": float(np.median(frag)),
        "dice_by_contrast": by_contrast,
        "misaligned_masks": misaligned,
        "latency_ms_median": float(np.median(latencies) * 1000), "device": str(device),
        "examples": {tag: {"frame": rows[i]["frame"], "dice": rows[i]["dice"]} for tag, i in picks.items()},
        "leakage": _leakage_report(),
    }
    (ROOT / "docs/vein_error_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
