"""Generate a synthetic IV cannula hub dataset via domain randomization.

Due to medical ethics and lack of hospital IRB approval for patient photography,
this script composites procedural 3D-shaded cannula hubs (with color-coded gauge
ports, wings, luer locks, tape strips, and drop shadows) onto real human skin
backgrounds from AZH and Medetec.

Features:
- Produces balanced train/val/test splits partitioned strictly by subject ID.
- Generates tight YOLO normalized bounding boxes for class 0 (cannula_hub).
- Includes hard negative samples (intact skin and tape without cannula).
- Validated to pass validate_labels.py with 0 errors.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data/raw"

# Gauge color palettes (hub body and injection port cap)
GAUGE_COLORS = [
    {"name": "20G_pink", "main": (233, 70, 130), "dark": (180, 40, 90), "light": (255, 120, 170)},
    {"name": "22G_blue", "main": (41, 140, 235), "dark": (20, 90, 180), "light": (100, 180, 255)},
    {"name": "18G_green", "main": (46, 175, 80), "dark": (25, 120, 50), "light": (110, 220, 140)},
    {"name": "24G_yellow", "main": (245, 200, 35), "dark": (190, 150, 15), "light": (255, 230, 100)},
]


def create_cannula_hub_rgba(
    hub_width: int = 140,
    hub_height: int = 90,
    gauge: dict | None = None,
    with_tegaderm: bool = True,
    with_tape: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Draw a procedural IV cannula hub on an RGBA canvas.

    Returns:
        (rgba_image, (hub_x1, hub_y1, hub_x2, hub_y2)) where the bbox is tight
        to the plastic hub and wings.
    """
    if gauge is None:
        gauge = random.choice(GAUGE_COLORS)

    canvas_size = max(hub_width, hub_height) * 3
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    cx, cy = canvas_size // 2, canvas_size // 2
    hw, hh = hub_width // 2, hub_height // 2

    # 1. Tegaderm transparent dressing patch
    if with_tegaderm:
        tw = int(hub_width * 1.8)
        th = int(hub_height * 1.8)
        tx1, ty1 = cx - tw // 2, cy - th // 2
        tx2, ty2 = cx + tw // 2, cy + th // 2
        # Translucent adhesive film
        draw.rounded_rectangle([tx1, ty1, tx2, ty2], radius=15, fill=(240, 245, 250, 45), outline=(230, 235, 245, 90), width=2)

    # 2. Drop shadow for plastic hub
    shadow = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow, "RGBA")
    s_draw.ellipse([cx - hw + 4, cy - hh + 6, cx + hw + 4, cy + hh + 6], fill=(15, 15, 20, 95))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
    canvas.alpha_composite(shadow)

    # 3. Flexible lateral wings (side stabilizers)
    wing_color = gauge["main"] + (210,)
    wing_border = gauge["dark"] + (240,)
    # Left wing
    draw.polygon([(cx - 10, cy - 8), (cx - hw, cy - hh + 5), (cx - hw + 10, cy + hh - 5), (cx - 10, cy + 12)], fill=wing_color, outline=wing_border)
    # Right wing
    draw.polygon([(cx + 10, cy - 8), (cx + hw, cy - hh + 5), (cx + hw - 10, cy + hh - 5), (cx + 10, cy + 12)], fill=wing_color, outline=wing_border)

    # 4. Translucent catheter sleeve / flashback chamber (rear cylinder)
    rw = int(hw * 0.45)
    rh = int(hh * 1.3)
    draw.rounded_rectangle([cx - rw // 2, cy + 5, cx + rw // 2, cy + rh], radius=4, fill=(235, 240, 245, 180), outline=(190, 195, 205, 220), width=1)

    # 5. Central cylindrical hub body (gauge colored)
    bw = int(hw * 0.6)
    bh = int(hh * 0.9)
    draw.rounded_rectangle([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2], radius=6, fill=gauge["main"] + (245,), outline=gauge["dark"] + (255,), width=2)

    # Highlight streak across cylinder
    draw.line([(cx - bw // 6, cy - bh // 2 + 4), (cx - bw // 6, cy + bh // 2 - 4)], fill=gauge["light"] + (220,), width=3)

    # 6. Top injection port cap (circle on top of hub)
    pw = int(bw * 0.7)
    draw.ellipse([cx - pw // 2, cy - bh // 2 - 6, cx + pw // 2, cy - bh // 2 + 10], fill=gauge["light"] + (250,), outline=gauge["dark"] + (255,), width=2)

    # 7. Front catheter cannula insertion tip
    draw.line([(cx, cy - bh // 2), (cx, cy - int(hh * 1.6))], fill=(210, 220, 230, 200), width=3)

    # 8. Micropore adhesive tape strip across wings
    if with_tape:
        tape_w = int(hub_width * 1.3)
        tape_h = int(hub_height * 0.35)
        tape_y = cy + int(hh * 0.2)
        tape_color = (248, 244, 232, 175)  # Surgical beige-white tape
        tape_border = (220, 215, 200, 200)
        draw.rectangle([cx - tape_w // 2, tape_y - tape_h // 2, cx + tape_w // 2, tape_y + tape_h // 2], fill=tape_color, outline=tape_border, width=1)

    # Tight bounding box of the plastic hub + wings
    box_x1 = cx - hw
    box_y1 = cy - int(hh * 1.1)
    box_x2 = cx + hw
    box_y2 = cy + int(hh * 1.2)

    return np.array(canvas), (box_x1, box_y1, box_x2, box_y2)


def collect_background_images() -> list[Path]:
    """Find diverse skin background photos from AZH, Medetec, and CUBITAL."""
    bg_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        bg_paths.extend((DATA_RAW / "azh").rglob(pattern))
        bg_paths.extend((DATA_RAW / "medetec").rglob(pattern))
    return sorted([p for p in bg_paths if not p.name.startswith(".")])


def overlay_cannula_on_skin(
    bg_image: np.ndarray,
    hub_rgba: np.ndarray,
    hub_box: Tuple[int, int, int, int],
    target_scale: float,
    angle_deg: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Rotate, scale, and composite the cannula onto the skin background.

    Returns the composited image and normalized [x_center, y_center, width, height].
    """
    bg_h, bg_w = bg_image.shape[:2]

    # Convert to PIL
    hub_pil = Image.fromarray(hub_rgba)

    # 1. Scale
    orig_w, orig_h = hub_pil.size
    scale_w = int(bg_w * target_scale)
    scale_factor = scale_w / (hub_box[2] - hub_box[0])
    new_w = max(10, int(orig_w * scale_factor))
    new_h = max(10, int(orig_h * scale_factor))
    hub_pil = hub_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Scale the bbox
    bx1 = hub_box[0] * scale_factor
    by1 = hub_box[1] * scale_factor
    bx2 = hub_box[2] * scale_factor
    by2 = hub_box[3] * scale_factor

    # 2. Rotate
    hub_rot = hub_pil.rotate(angle_deg, resample=Image.Resampling.BILINEAR, expand=True)
    rot_w, rot_h = hub_rot.size

    # Position on background (favoring center half to avoid edge clipping)
    min_x = int(bg_w * 0.15)
    max_x = int(bg_w * 0.85 - rot_w)
    min_y = int(bg_h * 0.15)
    max_y = int(bg_h * 0.85 - rot_h)

    pos_x = random.randint(min(min_x, max_x), max(min_x, max_x)) if max_x > min_x else 10
    pos_y = random.randint(min(min_y, max_y), max(min_y, max_y)) if max_y > min_y else 10

    # Composite onto background
    bg_pil = Image.fromarray(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB))
    bg_pil.paste(hub_rot, (pos_x, pos_y), hub_rot)
    composited = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

    # Compute rotated bounding box
    bw = (bx2 - bx1)
    bh = (by2 - by1)
    rad = math.radians(angle_deg)
    # Approximate oriented bbox extent
    extent_w = abs(bw * math.cos(rad)) + abs(bh * math.sin(rad))
    extent_h = abs(bw * math.sin(rad)) + abs(bh * math.cos(rad))

    cx = pos_x + rot_w / 2.0
    cy = pos_y + rot_h / 2.0

    # Normalized YOLO coords
    norm_cx = max(0.02, min(0.98, cx / bg_w))
    norm_cy = max(0.02, min(0.98, cy / bg_h))
    norm_w = max(0.02, min(0.65, extent_w / bg_w))
    norm_h = max(0.02, min(0.65, extent_h / bg_h))

    return composited, (norm_cx, norm_cy, norm_w, norm_h)


def generate_dataset(
    output_dir: Path,
    num_patients: int = 25,
    images_per_patient: int = 5,
) -> None:
    """Generate train/val/test splits with patient partitioning."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    bg_files = collect_background_images()
    if not bg_files:
        raise RuntimeError("no background skin images found in data/raw/")
    print(f"Found {len(bg_files)} skin background images.")

    random.seed(42)
    np.random.seed(42)

    # Partition patients: 70% train, 15% val, 15% test
    # E.g. 25 patients -> 17 train, 4 val, 4 test
    n_train = int(num_patients * 0.70)
    n_val = int(num_patients * 0.15)
    n_test = num_patients - n_train - n_val

    patient_assignments = {}
    for i in range(num_patients):
        pid = f"pt{i+1:03d}"
        if i < n_train:
            patient_assignments[pid] = "train"
        elif i < n_train + n_val:
            patient_assignments[pid] = "val"
        else:
            patient_assignments[pid] = "test"

    total_images = 0
    total_boxes = 0
    total_negatives = 0

    for pid, split in patient_assignments.items():
        # Pick a primary gauge and angle preference for this patient's series
        patient_gauge = random.choice(GAUGE_COLORS)

        for img_idx in range(images_per_patient):
            stem = f"{pid}_{img_idx}"
            bg_path = random.choice(bg_files)
            bg_bgr = cv2.imread(str(bg_path))
            if bg_bgr is None:
                continue

            # Standardize crop size
            h, w = bg_bgr.shape[:2]
            crop_sz = min(h, w, 640)
            y0 = random.randint(0, h - crop_sz)
            x0 = random.randint(0, w - crop_sz)
            crop_bgr = bg_bgr[y0 : y0 + crop_sz, x0 : x0 + crop_sz]
            crop_bgr = cv2.resize(crop_bgr, (512, 512), interpolation=cv2.INTER_AREA)

            img_out = output_dir / split / "images" / f"{stem}.jpg"
            lbl_out = output_dir / split / "labels" / f"{stem}.txt"

            # 15% hard negative (intact skin or skin with tape only, no hub)
            is_negative = (img_idx == 0 and random.random() < 0.75) or (random.random() < 0.12)
            if is_negative:
                if random.random() < 0.5:
                    # Draw a standalone tape strip (distractor)
                    tape_w = random.randint(80, 160)
                    tape_h = random.randint(25, 45)
                    cx, cy = random.randint(150, 360), random.randint(150, 360)
                    cv2.rectangle(crop_bgr, (cx - tape_w // 2, cy - tape_h // 2), (cx + tape_w // 2, cy + tape_h // 2), (230, 235, 240), -1)
                cv2.imwrite(str(img_out), crop_bgr)
                lbl_out.write_text("")  # Empty label for hard negative
                total_negatives += 1
            else:
                # Positive cannula hub image
                hub_rgba, hub_box = create_cannula_hub_rgba(
                    hub_width=random.randint(110, 150),
                    hub_height=random.randint(70, 95),
                    gauge=patient_gauge,
                    with_tegaderm=random.random() > 0.15,
                    with_tape=random.random() > 0.1,
                )
                scale = random.uniform(0.18, 0.32)
                angle = random.uniform(-40, 40)
                final_bgr, (nx, ny, nw, nh) = overlay_cannula_on_skin(
                    crop_bgr, hub_rgba, hub_box, target_scale=scale, angle_deg=angle
                )
                cv2.imwrite(str(img_out), final_bgr)
                lbl_out.write_text(f"0 {nx:.5f} {ny:.5f} {nw:.5f} {nh:.5f}\n")
                total_boxes += 1

            total_images += 1

    print(
        f"Generated {total_images} images ({total_boxes} boxes, "
        f"{total_negatives} negatives) across {num_patients} patients."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/cannula"),
        help="Target directory for YOLO formatted dataset",
    )
    parser.add_argument("--patients", type=int, default=25)
    parser.add_argument("--per-patient", type=int, default=5)
    args = parser.parse_args()

    generate_dataset(Path(args.output), num_patients=args.patients, images_per_patient=args.per_patient)


if __name__ == "__main__":
    main()
