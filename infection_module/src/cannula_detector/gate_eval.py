"""False-acceptance / false-rejection of the cannula gate, by data source.

    infection_module/.venv/bin/python -m infection_module.src.cannula_detector.gate_eval

Synthetic and real results are kept separate on purpose: the detector was
trained on 90 synthetic composites, so synthetic numbers say little about real
photos. Threshold alternatives are compared on the synthetic VALIDATION split
only; the deployed 0.5 gate is then reported on everything else. Writes
docs/cannula_gate_eval.json.

States, as in the API: confidence >= 0.50 verified; 0.25-0.50 "unable to
verify"; < 0.25 no cannula.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2

from infection_module.src.cannula_detector.inference import detect_cannula

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "infection_module/models/cannula_yolo11.pt"
GATE, POSSIBLE = 0.50, 0.25


def confidence(path: Path) -> float:
    found = detect_cannula(cv2.imread(str(path)), conf_threshold=0.05, weights=WEIGHTS)
    return found["confidence"] if found else 0.0


def labelled(split: str) -> tuple[list[float], list[float]]:
    pos, neg = [], []
    for image in sorted((ROOT / f"data/cannula/{split}/images").glob("*")):
        label = (ROOT / f"data/cannula/{split}/labels" / f"{image.stem}.txt")
        (pos if label.exists() and label.stat().st_size > 0 else neg).append(confidence(image))
    return pos, neg


def rates(pos: list[float], neg: list[float], gate: float) -> dict:
    return {
        "false_rejection": sum(c < gate for c in pos) / len(pos) if pos else None,
        "false_acceptance": sum(c >= gate for c in neg) / len(neg) if neg else None,
        "n_pos": len(pos), "n_neg": len(neg),
    }


def main() -> None:
    random.seed(0)
    val_pos, val_neg = labelled("val")
    test_pos, test_neg = labelled("test")

    negatives = {
        "cubital_nir_forearms": random.sample(sorted((ROOT / "data/raw/cubital/square_dataset/nir_images").glob("*.png")), 200),
        "azh_wounds": random.sample(sorted((ROOT / "data/raw/azh").rglob("*.*g")), 100),
        "medetec_wounds": random.sample(sorted((ROOT / "data/raw/medetec").rglob("*.jpg")), 100),
    }
    real_negative = {}
    for name, paths in negatives.items():
        confs = [confidence(p) for p in paths]
        real_negative[name] = {
            "n": len(confs),
            "false_acceptance_at_0.50": sum(c >= GATE for c in confs) / len(confs),
            "unable_to_verify_0.25_0.50": sum(POSSIBLE <= c < GATE for c in confs) / len(confs),
            "max_confidence": max(confs),
        }

    uploads = ROOT / "backend/data/uploads/patients/1/site"
    real_iv = {
        "iv_dressing_photo": uploads / "20261009/1ce83936409abdc6.png",
        "iv_hand_photo": uploads / "20261009/ac619862d2ffa641.png",
        "landscape_no_cannula": uploads / "20261014/1f616f6a726f9413.png",
    }
    real_photos = {name: round(confidence(p), 3) for name, p in real_iv.items() if p.exists()}

    summary = {
        "synthetic_val_threshold_sweep": {f"{g:.2f}": rates(val_pos, val_neg, g) for g in (0.3, 0.4, 0.5, 0.6, 0.7)},
        "synthetic_test_at_0.50": rates(test_pos, test_neg, GATE),
        "real_images_without_cannula": real_negative,
        "real_photos_confidence": real_photos,
        "note": "No labelled real cannula dataset exists for this project; real-photo "
                "false-rejection cannot be estimated beyond the individual photos listed.",
    }
    (ROOT / "docs/cannula_gate_eval.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
