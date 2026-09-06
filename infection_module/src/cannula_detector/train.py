"""Fine-tune YOLOv11 for cannula hub detection.

    python infection_module/src/cannula_detector/train.py

Requires a labelled dataset that does not exist publicly - see
docs/labeling_guide.md. Budget 1-2 days to produce it before running this.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def train(
    data_yaml: Path,
    output_path: Path,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
) -> Path:
    """Fine-tune from a COCO-pretrained checkpoint.

    Nano variant by default: with 100-200 training images, a larger backbone
    overfits before it generalises. `yolo11s.pt` is the fallback if nano
    underfits - check the val mAP curve in runs/ before reaching for it.
    """
    from ultralytics import YOLO

    if not data_yaml.exists():
        raise FileNotFoundError(f"no dataset config at {data_yaml}")

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(REPO_ROOT / "runs"),
        name="cannula_yolo11",
        exist_ok=True,
        # --- augmentation, tuned for a small dataset ---
        augment=True,
        mosaic=1.0,        # cheap diversity from 4-image composites
        fliplr=0.5,
        flipud=0.0,        # an arm photographed upside down is not a real case
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        # HSV jitter kept moderate on purpose. Aggressive hue/saturation shifts
        # would teach the detector to ignore redness - the exact cue the Phase 5
        # infection model depends on - so this must not be raised to "improve"
        # detection numbers in isolation.
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        # Disabled: mixup blends a cannula with an unrelated image, producing
        # translucent hubs that do not occur and diluting an already small set.
        mixup=0.0,
    )

    best = REPO_ROOT / "runs" / "cannula_yolo11" / "weights" / "best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_path)
    log.info("best weights -> %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent / "data.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "infection_module/models/cannula_yolo11.pt"),
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    train(
        Path(args.data), Path(args.output), args.model,
        args.epochs, args.imgsz, args.batch, args.patience,
    )


if __name__ == "__main__":
    main()
