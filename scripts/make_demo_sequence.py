"""Build a SIMULATED three-day IV-site sequence for demonstrating the monitor.

    infection_module/.venv/bin/python scripts/make_demo_sequence.py BACKGROUND.png OUT_DIR

There is no public series of real IV-site photos over several days, so this
makes one from a single real limb photo:

* a procedural cannula hub (the same renderer the detector was trained on) is
  composited onto the skin at a fixed spot,
* day 1 is that photo unchanged,
* days 2 and 3 add CIELAB a* redness in a Gaussian spreading from the hub.

Every file name says SIMULATED. These images demonstrate that the pipeline
detects the hub, measures redness and raises an alert as it grows. They are
not evidence that the system detects real phlebitis.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infection_module.src.cannula_detector.generate_dataset import (  # noqa: E402
    GAUGE_COLORS, create_cannula_hub_rgba,
)

DAYS = [(1, 0.0, "baseline"), (2, 9.0, "mild"), (3, 18.0, "spreading")]

# Read by the API (app/core/ml.py::sample_origin) so a result from these files
# is labelled "simulated demo sample" regardless of what the file is renamed to.
SIMULATED_KEY = "IntelliCannula-Simulated"


def composite_hub(bg: np.ndarray, centre: tuple[int, int], width_frac: float) -> tuple[np.ndarray, list[float]]:
    random.seed(7)
    rgba, (hx1, hy1, hx2, hy2) = create_cannula_hub_rgba(
        hub_width=130, hub_height=85, gauge=GAUGE_COLORS[1], with_tegaderm=True, with_tape=True,
    )
    hub = Image.fromarray(rgba)
    scale = bg.shape[1] * width_frac / (hx2 - hx1)
    hub = hub.resize((int(hub.width * scale), int(hub.height * scale)), Image.Resampling.BILINEAR)
    px, py = int(centre[0] - hub.width / 2), int(centre[1] - hub.height / 2)
    canvas = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    canvas.paste(hub, (px, py), hub)
    box = [px + hx1 * scale, py + hy1 * scale, px + hx2 * scale, py + hy2 * scale]
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR), box


def redden(img: np.ndarray, centre: tuple[int, int], amount: float, sigma: float,
           skin_only: bool = True) -> np.ndarray:
    """Add a* redness spreading from the hub. Erythema is in skin, so by default
    tape, dressing, plastic and sheets are left unchanged."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    g = np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * sigma**2))
    if skin_only:
        from infection_module.src.erythema import skin_mask
        g *= cv2.GaussianBlur(skin_mask(img).astype(np.float32), (7, 7), 0)
    lab[..., 1] += amount * g
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background")
    parser.add_argument("out_dir")
    parser.add_argument("--centre", type=int, nargs=2, default=[300, 700],
                        help="hub centre in background pixels (on visible skin)")
    parser.add_argument("--width-frac", type=float, default=0.12)
    args = parser.parse_args()

    bg = cv2.imread(args.background)
    if bg is None:
        raise SystemExit(f"cannot read {args.background}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    centre = tuple(args.centre)
    day1, box = composite_hub(bg, centre, args.width_frac)
    sigma = 1.6 * (box[2] - box[0])
    for day, amount, tag in DAYS:
        img = day1 if amount == 0 else redden(day1, centre, amount, sigma, skin_only=True)
        path = out / f"SIMULATED_day{day}_{tag}.png"
        meta = PngInfo()
        meta.add_text(SIMULATED_KEY, "true")
        meta.add_text("Description", f"SIMULATED day {day}: procedural cannula hub; "
                      f"{amount:g} CIELAB a* units of artificial redness added. Not patient data.")
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(path, pnginfo=meta)
        print(path)


if __name__ == "__main__":
    main()
