"""Wound classification dataset: raw AZH / Medetec -> processed splits.

Run from repo root:

    python infection_module/src/data_loader.py --prepare

Label provenance - read before trusting any metric built on this
-----------------------------------------------------------------
AZH labels are *wound type*, not infection status:

    D  diabetic      P  pressure     V  venous
    S  surgical      N  normal skin  BG background

There is no ground-truth infected/uninfected label in any public dataset used
here. The binary target this module produces is therefore a PROXY:

    is_infected = 1  for wound tissue (D, P, V, S)
    is_infected = 0  for normal skin  (N)
    BG is dropped entirely - it is neither.

What that proxy actually learns is "wound tissue vs intact skin", which is a
weaker claim than "infected vs uninfected". It is still the right first target
for IntelliCannula, because an IV site developing phlebitis moves from intact
skin toward inflamed tissue, and that is the transition we need to detect. But
every downstream metric must be reported as proxy-label performance. Phase 7's
limitations section says so explicitly.

Medetec is never mixed into the AZH split. It is written separately as an
external generalization test set: different cameras, clinicians, and decade, so
performance there is the honest estimate of transfer.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

AZH_CLASS_NAMES = {
    "D": "diabetic",
    "P": "pressure",
    "V": "venous",
    "S": "surgical",
    "N": "normal",
    "BG": "background",
}

# Wound classes that map to the positive proxy label. See module docstring.
WOUND_CLASSES = {"diabetic", "pressure", "venous", "surgical"}
HEALTHY_CLASSES = {"normal"}
EXCLUDED_CLASSES = {"background"}


class WoundClassificationDataset(Dataset):
    """Images from data/processed/infection/{split}/<class>/.

    Parameters
    ----------
    label_mode :
        "binary" -> `is_infected` proxy target (wound=1, normal=0)
        "type"   -> the multi-class wound type index

    Returns (image CHW float32 in [0,1], label int).
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transform=None,
        label_mode: str = "binary",
        image_size: int = 224,
    ):
        if label_mode not in {"binary", "type"}:
            raise ValueError(f"label_mode must be 'binary' or 'type', got {label_mode!r}")

        self.root = Path(root_dir) / split
        self.transform = transform
        self.label_mode = label_mode
        self.image_size = image_size

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.root} not found - run "
                "`python infection_module/src/data_loader.py --prepare` first"
            )

        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self._label_for(class_dir.name)
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"no images found under {self.root}")
        log.info(
            "%s split: %d images across %d classes", split, len(self.samples), len(self.classes)
        )

    def _label_for(self, class_name: str) -> int:
        if self.label_mode == "binary":
            return 1 if class_name in WOUND_CLASSES else 0
        return self.class_to_idx[class_name]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        path, label = self.samples[idx]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))

        image = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        return image, label


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


def _collect_azh(azh_dir: Path) -> dict[str, list[Path]]:
    """Gather AZH images keyed by human-readable class name.

    AZH ships pre-split Train/ and Test/ trees with single-letter class dirs.
    Both are pooled here and re-split, because the shipped split is
    image-level; re-splitting lets us control the ratio and record a manifest.
    """
    by_class: dict[str, list[Path]] = defaultdict(list)
    if not azh_dir.is_dir():
        log.warning("AZH directory %s not found, skipping", azh_dir)
        return by_class

    for split_dir in azh_dir.rglob("*"):
        if not split_dir.is_dir() or split_dir.name not in AZH_CLASS_NAMES:
            continue
        class_name = AZH_CLASS_NAMES[split_dir.name]
        for path in split_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                by_class[class_name].append(path)

    for name, paths in sorted(by_class.items()):
        log.info("AZH %-11s %d images", name, len(paths))
    return by_class


def _collect_medetec(medetec_dir: Path) -> dict[str, list[Path]]:
    """Gather Medetec wound images keyed by gallery category.

    The site layout is:

        slide scans/<category>/index.html
        slide scans/<category>/thumbnails/<name>.jpg    ~8 KB previews
        slide scans/<category>/images/<name>.jpg        full resolution

    Three things this has to get right:

    1. Prefer `images/` over `thumbnails/`. An 8 KB preview has been through
       aggressive JPEG quantisation, which destroys exactly the fine colour and
       texture detail the Phase 5 features measure. A thumbnail is only used
       when no full-size counterpart was downloaded.
    2. Skip `slide scans dressings/` - those are product photographs of dressing
       materials, not wounds.
    3. Skip the site's own `images/` and `files/` folders at the root, which
       hold logos and page furniture.

    Medetec has no labels compatible with AZH, so the gallery name is kept as
    the category. `extravasation-wound-images` is the one that matters most:
    extravasation is an IV-infiltration injury, the closest public imagery to
    IntelliCannula's actual failure mode.
    """
    by_class: dict[str, list[Path]] = defaultdict(list)
    if not medetec_dir.is_dir():
        log.warning("Medetec directory %s not found, skipping", medetec_dir)
        return by_class

    gallery_root = medetec_dir / "slide scans"
    if not gallery_root.is_dir():
        log.warning("no 'slide scans' tree under %s - was the crawl interrupted?", medetec_dir)
        return by_class

    full_size: dict[str, dict[str, Path]] = defaultdict(dict)
    thumbnails: dict[str, dict[str, Path]] = defaultdict(dict)

    for path in gallery_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            relative = path.relative_to(gallery_root)
        except ValueError:  # pragma: no cover - rglob guarantees containment
            continue
        if len(relative.parts) < 2:
            continue

        category = relative.parts[0].lower().replace(" ", "_")
        bucket = thumbnails if "thumbnails" in {p.lower() for p in relative.parts} else full_size
        bucket[category][path.stem] = path

    for category in sorted(set(full_size) | set(thumbnails)):
        chosen = dict(thumbnails.get(category, {}))
        chosen.update(full_size.get(category, {}))  # full size wins on stem collision
        by_class[category] = sorted(chosen.values())

    n_thumb_only = sum(
        1
        for category in by_class
        for stem in thumbnails.get(category, {})
        if stem not in full_size.get(category, {})
    )
    if n_thumb_only:
        log.warning(
            "%d Medetec images are thumbnail-only (no full-size counterpart downloaded); "
            "re-run the crawler to completion for better colour fidelity",
            n_thumb_only,
        )

    for name, paths in sorted(by_class.items()):
        log.info("Medetec %-32s %d images", name, len(paths))
    return by_class


def _stratified_split(
    paths: list[Path], val_frac: float, test_frac: float, rng: random.Random
) -> dict[str, list[Path]]:
    """Split one class's images into train/val/test.

    Called per class, so the overall split is stratified by construction.
    AZH has no subject IDs in filenames, so this is necessarily image-level -
    unlike the vein module, which splits by subject. Flagged as a limitation:
    if two photos of the same wound exist, they may straddle the split.
    """
    shuffled = sorted(paths)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def prepare_infection_dataset(
    azh_dir: str | Path,
    medetec_dir: str | Path,
    output_dir: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Build data/processed/infection/{train,val,test}/ + medetec_external_test/.

    Returns nested counts {split: {class: n}}.
    """
    azh_dir, medetec_dir, output_dir = Path(azh_dir), Path(medetec_dir), Path(output_dir)
    rng = random.Random(seed)

    azh = _collect_azh(azh_dir)
    if not azh:
        raise RuntimeError(
            "no AZH images found. Download first: "
            "python scripts/download_datasets.py --dataset azh"
        )

    counts: dict[str, dict[str, int]] = defaultdict(dict)
    manifest_files: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for class_name, paths in sorted(azh.items()):
        if class_name in EXCLUDED_CLASSES:
            log.info("dropping %d '%s' images (neither wound nor skin)", len(paths), class_name)
            continue

        splits = _stratified_split(paths, val_frac, test_frac, rng)
        for split, split_paths in splits.items():
            dest_dir = output_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for path in split_paths:
                shutil.copy2(path, dest_dir / path.name)
            counts[split][class_name] = len(split_paths)
            manifest_files[split][class_name] = [p.name for p in split_paths]

    # Medetec: held out whole, never split, never mixed into training.
    medetec = _collect_medetec(medetec_dir)
    external_root = output_dir / "medetec_external_test"
    for category, paths in sorted(medetec.items()):
        dest_dir = external_root / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, dest_dir / path.name)
        counts["medetec_external_test"][category] = len(paths)

    proxy_totals = {
        split: {
            "positive_wound": sum(
                n for c, n in class_counts.items() if c in WOUND_CLASSES
            ),
            "negative_skin": sum(
                n for c, n in class_counts.items() if c in HEALTHY_CLASSES
            ),
        }
        for split, class_counts in counts.items()
        if split != "medetec_external_test"
    }

    manifest = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "split_level": "image",
        "split_level_caveat": (
            "AZH filenames carry no subject IDs, so this split is image-level. "
            "Repeat photographs of one wound may straddle splits."
        ),
        "label_provenance": (
            "PROXY labels. AZH provides wound type, not infection status. "
            "is_infected=1 for {diabetic,pressure,venous,surgical}, 0 for {normal}, "
            "background dropped. Report all metrics as proxy-label performance."
        ),
        "counts": {k: dict(v) for k, v in counts.items()},
        "proxy_binary_totals": proxy_totals,
        "files": {k: dict(v) for k, v in manifest_files.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for split in ("train", "val", "test", "medetec_external_test"):
        if split in counts:
            total = sum(counts[split].values())
            log.info("%-24s %d images %s", split, total, dict(sorted(counts[split].items())))
    log.info("proxy binary totals: %s", proxy_totals)
    log.info("manifest -> %s", manifest_path)
    return {k: dict(v) for k, v in counts.items()}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="build processed splits")
    parser.add_argument("--azh-dir", default=str(repo_root / "data/raw/azh"))
    parser.add_argument("--medetec-dir", default=str(repo_root / "data/raw/medetec"))
    parser.add_argument("--output-dir", default=str(repo_root / "data/processed/infection"))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.prepare:
        parser.error("nothing to do - pass --prepare")

    counts = prepare_infection_dataset(
        args.azh_dir,
        args.medetec_dir,
        args.output_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    for split, class_counts in counts.items():
        print(f"{split:>24}: {sum(class_counts.values())} images")


if __name__ == "__main__":
    main()
