#!/usr/bin/env python3
"""Fetch the real public datasets into data/raw/.

Phase 1 ships acquisition only - no parsing, no splitting, no synthetic data.
Phase 2 (data pipeline) consumes what this writes.

    python scripts/download_datasets.py --dataset cubital
    python scripts/download_datasets.py --dataset azh
    python scripts/download_datasets.py --dataset medetec
    python scripts/download_datasets.py --dataset all

U-DAVIS is deliberately absent: it has no public mirror and must be requested
from the authors (see docs/dataset_sources.md).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

CUBITAL_DRIVE_FOLDER = "19DaK7T81qTxgBzirvBGdMyVUj8mOCcOx"
AZH_REPO = "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations.git"
MEDETEC_INDEX = "https://www.medetec.co.uk/files/medetec-image-databases.html"


def _require(binary: str, install_hint: str) -> None:
    if shutil.which(binary) is None:
        sys.exit(f"error: `{binary}` not found on PATH. Install it with: {install_hint}")


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_cubital() -> None:
    """CUBITAL: 2,016 NIR antecubital-fossa images + masks + nurse annotations.

    Hosted on Google Drive, so this needs `gdown`. The folder is large; expect
    a multi-GB download. Version C (512x512) is the one Phase 2 defaults to,
    but the whole folder is pulled so version A stays available.
    """
    _require("gdown", "uv tool install gdown")
    out = RAW_DIR / "cubital"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cubital] -> {out}")
    _run(["gdown", "--folder", "--remaining-ok", "-O", str(out), CUBITAL_DRIVE_FOLDER])


def download_azh() -> None:
    """AZH wound dataset: 730 ROI + 538 whole-wound RGB images, 4 wound types.

    Distributed inside a git repo, so a shallow clone is the whole job.
    """
    _require("git", "xcode-select --install")
    out = RAW_DIR / "azh"
    if out.exists():
        print(f"[azh] already present at {out}, skipping")
        return
    print(f"[azh] -> {out}")
    _run(["git", "clone", "--depth", "1", AZH_REPO, str(out)])


def download_medetec() -> None:
    """Medetec: ~594 RGB wound images across ~15 browsable category galleries.

    There is no archive to fetch - the images live on per-category HTML pages,
    so this mirrors the image files only, staying inside the slide-scan tree.
    """
    _require("wget", "brew install wget")
    out = RAW_DIR / "medetec"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[medetec] -> {out}")
    _run([
        "wget",
        "--recursive",
        "--level=3",
        "--no-parent",
        "--accept=jpg,jpeg,png,gif",
        "--wait=1",              # be polite to a small static host
        "--random-wait",
        "--no-host-directories",
        "--directory-prefix", str(out),
        MEDETEC_INDEX,
    ])


DOWNLOADERS = {
    "cubital": download_cubital,
    "azh": download_azh,
    "medetec": download_medetec,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DOWNLOADERS, "all"],
        required=True,
        help="which dataset to fetch into data/raw/",
    )
    args = parser.parse_args()

    targets = list(DOWNLOADERS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        DOWNLOADERS[name]()
    print("\ndone. U-DAVIS must be requested separately - see docs/dataset_sources.md")


if __name__ == "__main__":
    main()
