"""Inference-time post-processing experiments on the VALIDATION split.

    infection_module/.venv/bin/python -m vein_module.src.postprocess_experiments

Model selection only - the test split is never read here. Writes
docs/vein_postprocess_experiments.json. Compares, on every val image:
threshold grid, horizontal-flip test-time augmentation (TTA), and hysteresis
thresholding, each with the 60 px speckle filter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vein_module.src.augmentations import get_eval_transform
from vein_module.src.evaluate import segmentation_metrics
from vein_module.src.model import AttentionUNet
from vein_module.src.postprocess import binarize, hysteresis
from vein_module.src.train import get_device

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    "thr0.90_min60 (current)": dict(tta=False, kind="thr", high=0.90),
    "thr0.85_min60": dict(tta=False, kind="thr", high=0.85),
    "thr0.95_min60": dict(tta=False, kind="thr", high=0.95),
    "tta_thr0.90_min60": dict(tta=True, kind="thr", high=0.90),
    "hyst0.90/0.50_min60": dict(tta=False, kind="hyst", high=0.90, low=0.50),
    "hyst0.90/0.70_min60": dict(tta=False, kind="hyst", high=0.90, low=0.70),
    "tta_hyst0.90/0.70_min60": dict(tta=True, kind="hyst", high=0.90, low=0.70),
}


def main() -> None:
    device = get_device("auto")
    ckpt = torch.load(ROOT / "vein_module/models/attention_unet_best.pt", map_location=device, weights_only=False)
    model = AttentionUNet(base_channels=ckpt["config"].get("base_channels", 32)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = get_eval_transform(512)

    results = {name: [] for name in CONFIGS}
    timing = {"plain": [], "tta": []}
    images = sorted((ROOT / "data/processed/vein/val/images").glob("*.png"))
    for path in images:
        x = transform(image=cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))["image"]
        t = torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        with torch.inference_mode():
            start = time.perf_counter()
            plain = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            timing["plain"].append(time.perf_counter() - start)
            start = time.perf_counter()
            flipped = torch.sigmoid(model(torch.flip(t, dims=[3])))[0, 0].cpu().numpy()[:, ::-1]
            timing["tta"].append(time.perf_counter() - start + timing["plain"][-1])
        tta = (plain + flipped) / 2
        target = cv2.resize(cv2.imread(str(path).replace("/images/", "/masks/"), 0), (512, 512),
                            interpolation=cv2.INTER_NEAREST) > 127
        if target.sum() == 0:
            continue
        for name, cfg in CONFIGS.items():
            probs = tta if cfg["tta"] else plain
            pred = (binarize(probs, cfg["high"], 60) if cfg["kind"] == "thr"
                    else hysteresis(probs, cfg["high"], cfg["low"], 60))
            results[name].append(segmentation_metrics(pred, target))

    summary = {
        "split": "val", "checkpoint_epoch": ckpt["epoch"], "n_images_with_vein": len(results[next(iter(CONFIGS))]),
        "latency_ms_median": {k: float(np.median(v) * 1000) for k, v in timing.items()},
        "device": str(device),
        "configs": {
            name: {m: float(np.mean([r[m] for r in rows])) for m in ("dice", "iou", "precision", "recall")}
            for name, rows in results.items()
        },
    }
    out = ROOT / "docs/vein_postprocess_experiments.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
