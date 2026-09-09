#!/usr/bin/env python3
"""Validate a hand-labelled cannula dataset before training on it.

    python infection_module/src/cannula_detector/validate_labels.py --root data/cannula

Labelling 100-200 images is hours of work, and every failure mode below is one
that silently produces a bad model rather than an error: an unpaired label, a
box in pixel coordinates instead of normalised, a patient whose photos straddle
the train/test split. Catching them here costs seconds; catching them after
training costs the whole run.

Exits non-zero if anything is wrong, so it can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


class Issue:
    def __init__(self, level: str, message: str):
        self.level = level
        self.message = message

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


def _patient_id(stem: str) -> str:
    """Patient identifier from a filename, by convention `<patient>_<n>`.

    Splitting by image instead of by patient is the single most common way to
    inflate a small medical dataset's test score: three photos of one site land
    in train and test together and the model is graded on what it memorised.
    The same trap was found in AZH during Phase 2.
    """
    return stem.split("_")[0]


def validate_split(root: Path, split: str) -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"

    if not images_dir.is_dir():
        return [Issue("ERROR", f"missing directory {images_dir}")], {}

    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in labels_dir.iterdir() if p.suffix == ".txt"} if labels_dir.is_dir() else {}

    for stem in sorted(images.keys() - labels.keys()):
        issues.append(Issue(
            "ERROR",
            f"{split}: {images[stem].name} has no label file. If the image "
            "genuinely contains no cannula, create an EMPTY .txt - a missing "
            "file is ambiguous between 'no object' and 'not yet labelled'.",
        ))
    for stem in sorted(labels.keys() - images.keys()):
        issues.append(Issue("ERROR", f"{split}: {labels[stem].name} has no matching image"))

    n_boxes = n_empty = 0
    for stem, label_path in sorted(labels.items()):
        lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
        if not lines:
            n_empty += 1
            continue

        for lineno, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                issues.append(Issue(
                    "ERROR",
                    f"{split}/{label_path.name}:{lineno}: expected 5 fields "
                    f"(class x y w h), got {len(parts)}",
                ))
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                issues.append(Issue("ERROR", f"{split}/{label_path.name}:{lineno}: non-numeric field"))
                continue

            if class_id != 0:
                issues.append(Issue(
                    "ERROR",
                    f"{split}/{label_path.name}:{lineno}: class id {class_id}, "
                    "expected 0 (cannula_hub) - this is a single-class dataset",
                ))

            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                issues.append(Issue(
                    "ERROR",
                    f"{split}/{label_path.name}:{lineno}: coordinates "
                    f"({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}) outside [0,1]. YOLO "
                    "wants values normalised by image size, not pixels.",
                ))
                continue

            if w <= 0 or h <= 0:
                issues.append(Issue("ERROR", f"{split}/{label_path.name}:{lineno}: zero-area box"))
            elif w > 0.9 or h > 0.9:
                issues.append(Issue(
                    "WARN",
                    f"{split}/{label_path.name}:{lineno}: box covers "
                    f"{w * 100:.0f}%x{h * 100:.0f}% of the frame. Draw a tight "
                    "box on the hub, not the whole dressing - crop_roi expands "
                    "it by 40% per side and would pull in background.",
                ))
            elif w < 0.01 or h < 0.01:
                issues.append(Issue("WARN", f"{split}/{label_path.name}:{lineno}: box under 1% of frame"))

            n_boxes += 1

    return issues, {
        "images": len(images),
        "labels": len(labels),
        "boxes": n_boxes,
        "empty_labels": n_empty,
        "patients": {_patient_id(s) for s in images},
    }


def validate(root: Path) -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    stats: dict[str, dict] = {}

    for split in SPLITS:
        split_issues, split_stats = validate_split(root, split)
        issues.extend(split_issues)
        if split_stats:
            stats[split] = split_stats

    # Patient-level leakage across splits.
    for a in SPLITS:
        for b in SPLITS:
            if a >= b or a not in stats or b not in stats:
                continue
            shared = stats[a]["patients"] & stats[b]["patients"]
            if shared:
                issues.append(Issue(
                    "ERROR",
                    f"patient leakage: {sorted(shared)[:5]} appear in both "
                    f"{a} and {b}. Split by patient, not by image, or the test "
                    "mAP measures memorisation.",
                ))

    total_images = sum(s["images"] for s in stats.values())
    if total_images < 100:
        issues.append(Issue(
            "WARN",
            f"only {total_images} images. Below ~100 even YOLO11n overfits; "
            "the labelling guide targets 100-200.",
        ))

    total_boxes = sum(s["boxes"] for s in stats.values())
    if total_boxes and total_images:
        empty = sum(s["empty_labels"] for s in stats.values())
        if empty == 0:
            issues.append(Issue(
                "WARN",
                "no negative images (empty label files). Without hard negatives "
                "- sites with no cannula, watches, tape - the detector learns "
                "'find the shiny plastic thing' and fires on a wristwatch.",
            ))

    return issues, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3] / "data/cannula"),
        help="dataset root containing train/ val/ test/",
    )
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(
            f"no dataset at {root}\n"
            "Label images first - see docs/labeling_guide.md - then arrange them as:\n"
            "  data/cannula/{train,val,test}/{images,labels}/"
        )

    issues, stats = validate(root)

    print(f"dataset: {root}\n")
    for split, s in stats.items():
        print(
            f"  {split:<6} {s['images']:>4} images  {s['boxes']:>4} boxes  "
            f"{s['empty_labels']:>3} negatives  {len(s['patients']):>3} patients"
        )
    print()

    errors = [i for i in issues if i.level == "ERROR"]
    warnings = [i for i in issues if i.level == "WARN"]
    for issue in issues:
        print(f"  {issue}")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors or (args.strict and warnings):
        sys.exit(1)
    print("dataset looks ready for training")


if __name__ == "__main__":
    main()
