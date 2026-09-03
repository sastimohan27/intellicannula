"""Vein segmentation dataset: raw CUBITAL / U-DAVIS -> processed splits.

Run from repo root:

    python vein_module/src/data_loader.py --prepare

Design note - subject-level splitting. CUBITAL contributes two images per
subject (one per arm) and U-DAVIS has multiple frames per participant. An
image-level split would put a subject's left arm in train and their right arm
in test, which inflates Dice by memorising skin texture rather than learning
vein structure. Splits are therefore assigned by subject ID.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_SIZE = 512
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# CUBITAL masks are 3-level label maps, NOT binary: 0=background, 1=arm, 2=vein.
# Thresholding these at 127 yields an all-zero mask, so the vein class is
# selected explicitly. See docs/dataset_sources.md.
CUBITAL_VEIN_LABEL = 2

# U-DAVIS ships conventional 0/255 binary masks.
BINARY_THRESHOLD = 127


class VeinSegmentationDataset(Dataset):
    """Image/mask pairs from data/processed/vein/{split}/.

    Images and masks are matched by filename stem, so `s0042_left.png` in
    images/ pairs with `s0042_left.png` in masks/ regardless of extension.

    Returns
    -------
    image : float32 tensor-ready array, CHW, scaled to [0, 1]
    mask  : float32 array, 1HW, values in {0.0, 1.0}
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transform=None,
        image_size: int = IMAGE_SIZE,
    ):
        self.root = Path(root_dir) / split
        self.split = split
        self.transform = transform
        self.image_size = image_size

        image_dir = self.root / "images"
        mask_dir = self.root / "masks"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"{image_dir} not found - run `python vein_module/src/data_loader.py --prepare` first"
            )

        masks_by_stem = {
            p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        }
        self.pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            mask_path = masks_by_stem.get(img_path.stem)
            if mask_path is None:
                log.warning("no mask for %s, skipping", img_path.name)
                continue
            self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise RuntimeError(f"no image/mask pairs found in {self.root}")
        log.info("%s split: %d pairs", split, len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_path}")

        # Masks are already binarised during prepare(); this guards against a
        # hand-dropped file that never went through the pipeline.
        mask = (mask > BINARY_THRESHOLD).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(
                mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
            )

        image = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        mask = mask.astype(np.float32)[None, ...]
        return image, mask


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


@dataclass
class Sample:
    """One raw image/mask pair, tagged with the subject it came from."""

    image: Path
    mask: Path
    subject: str
    source: str  # "cubital" | "udavis"


def parse_subject_id(filename: str, source: str) -> str:
    """Derive a subject ID from a filename.

    CUBITAL names encode subject and arm, e.g. `0042_left.png` or `s0042_R.png`,
    so the leading numeric run identifies the person. U-DAVIS frames are
    similar. If nothing numeric is present the stem itself becomes the ID,
    which degrades safely to image-level splitting for that file rather than
    silently grouping unrelated images together.
    """
    stem = Path(filename).stem
    match = re.match(r"^[A-Za-z]*?(\d+)", stem)
    subject = match.group(1) if match else stem
    return f"{source}_{subject}"


def _binarize(mask: np.ndarray, source: str) -> np.ndarray:
    """Convert a source-specific mask into a 0/255 binary vein mask."""
    if source == "cubital":
        # Label map: keep only the vein class. Some CUBITAL versions ship the
        # same map scaled to 0/128/255, so accept both encodings.
        if mask.max() > 2:
            return ((mask > 200)).astype(np.uint8) * 255
        return (mask == CUBITAL_VEIN_LABEL).astype(np.uint8) * 255
    return (mask > BINARY_THRESHOLD).astype(np.uint8) * 255


def _find_pairs(root: Path, source: str) -> list[Sample]:
    """Locate image/mask pairs under a raw dataset directory.

    Datasets differ in layout, so this matches on directory name containing
    'mask' rather than assuming a fixed structure.
    """
    if not root.is_dir():
        log.warning("%s directory %s not found, skipping", source, root)
        return []

    mask_files: dict[str, Path] = {}
    image_files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # Only inspect the path *below* the dataset root. Using absolute parts
        # would misclassify every file whenever an ancestor directory happens to
        # contain "mask" (e.g. ~/masters-project/), silently yielding zero pairs.
        parts = {part.lower() for part in path.relative_to(root).parts}
        is_mask = any(
            "mask" in part or "label" in part or "annotation" in part for part in parts
        )
        target = mask_files if is_mask else image_files
        # Prefer the first match; later duplicates across versions are ignored.
        target.setdefault(path.stem, path)

    samples = [
        Sample(
            image=image_files[stem],
            mask=mask_files[stem],
            subject=parse_subject_id(stem, source),
            source=source,
        )
        for stem in sorted(image_files.keys() & mask_files.keys())
    ]
    log.info("%s: %d paired samples from %s", source, len(samples), root)
    return samples


def _split_subjects(
    subjects: list[str], val_frac: float, test_frac: float, seed: int
) -> dict[str, str]:
    """Assign each subject to exactly one split."""
    rng = random.Random(seed)
    shuffled = sorted(subjects)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))

    assignment = {}
    for i, subject in enumerate(shuffled):
        if i < n_test:
            assignment[subject] = "test"
        elif i < n_test + n_val:
            assignment[subject] = "val"
        else:
            assignment[subject] = "train"
    return assignment


def prepare_vein_dataset(
    cubital_dir: str | Path,
    udavis_dir: str | Path,
    output_dir: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> dict[str, int]:
    """Build data/processed/vein/{train,val,test}/{images,masks}/.

    U-DAVIS is optional - it has no public mirror and must be requested from
    its authors, so a missing directory logs a warning and the function
    proceeds on CUBITAL alone.

    Returns per-split image counts.
    """
    output_dir = Path(output_dir)

    samples = _find_pairs(Path(cubital_dir), "cubital") + _find_pairs(
        Path(udavis_dir), "udavis"
    )
    if not samples:
        raise RuntimeError(
            "no samples found. Download the datasets first: "
            "python scripts/download_datasets.py --dataset cubital"
        )

    by_subject: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_subject[sample.subject].append(sample)

    assignment = _split_subjects(
        list(by_subject.keys()), val_frac, test_frac, seed
    )

    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for subject, subject_samples in by_subject.items():
        split = assignment[subject]
        for sample in subject_samples:
            image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                log.warning("unreadable pair %s, skipping", sample.image.name)
                continue

            mask = _binarize(mask, sample.source)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            # Nearest-neighbour keeps the mask binary; bilinear would create
            # grey edge pixels that then get re-thresholded inconsistently.
            mask = cv2.resize(
                mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST
            )

            stem = f"{sample.source}_{sample.image.stem}"
            cv2.imwrite(str(output_dir / split / "images" / f"{stem}.png"), image)
            cv2.imwrite(str(output_dir / split / "masks" / f"{stem}.png"), mask)
            counts[split] += 1

    manifest = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "image_size": image_size,
        "split_level": "subject",
        "counts": counts,
        "n_subjects": {
            split: sum(1 for s, sp in assignment.items() if sp == split)
            for split in ("train", "val", "test")
        },
        "subjects": {
            split: sorted(s for s, sp in assignment.items() if sp == split)
            for split in ("train", "val", "test")
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info("counts: %s", counts)
    log.info("subjects: %s", manifest["n_subjects"])
    log.info("manifest -> %s", manifest_path)
    return counts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="build processed splits")
    parser.add_argument("--cubital-dir", default=str(repo_root / "data/raw/cubital"))
    parser.add_argument("--udavis-dir", default=str(repo_root / "data/raw/udavis"))
    parser.add_argument("--output-dir", default=str(repo_root / "data/processed/vein"))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.prepare:
        parser.error("nothing to do - pass --prepare")

    counts = prepare_vein_dataset(
        args.cubital_dir,
        args.udavis_dir,
        args.output_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    for split, n in counts.items():
        print(f"{split:>6}: {n} images")


if __name__ == "__main__":
    main()
