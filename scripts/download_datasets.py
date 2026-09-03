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
import os
import shutil
import subprocess
import sys
import time
import ssl
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

CUBITAL_DRIVE_FOLDER = "19DaK7T81qTxgBzirvBGdMyVUj8mOCcOx"
AZH_REPO = "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations.git"
MEDETEC_INDEX = "https://www.medetec.co.uk/files/medetec-image-databases.html"
MEDETEC_ROOT = "https://www.medetec.co.uk/"
MEDETEC_DELAY_S = 0.5           # be polite to a small static host
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif"}
USER_AGENT = "IntelliCannula-research-crawler/0.1 (academic use)"

_SSL_CTX: ssl.SSLContext | None = None


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
    # gdown >=5 dropped --remaining-ok; --continue makes reruns resumable instead.
    _run(["gdown", "--folder", "--continue", "-O", str(out), CUBITAL_DRIVE_FOLDER])


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


def _ssl_context() -> ssl.SSLContext:
    """TLS context backed by certifi.

    The python.org macOS build ships no CA bundle of its own, so urllib's
    default context fails with "unable to get local issuer certificate" on
    hosts that curl and openssl verify fine. Point it at certifi explicitly.

    Note: if your network runs a TLS-inspecting firewall (this project was
    developed behind a Fortinet appliance that re-signs some domains), that
    CA will not be in certifi either. Export the appliance's CA and set
    SSL_CERT_FILE to a bundle containing it - do not disable verification.
    """
    cafile = os.environ.get("SSL_CERT_FILE")
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _encode_url(url: str) -> str:
    """Percent-encode the path so spaces survive.

    Medetec paths contain literal spaces ("/slide scans/burns/"), which
    urllib rejects as control characters. safe= keeps the separators that
    are already meaningful.
    """
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        parts._replace(
            path=urllib.parse.quote(parts.path, safe="/%"),
            query=urllib.parse.quote(parts.query, safe="=&%"),
        )
    )


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(_encode_url(url), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
        return resp.read()


class _LinkParser(HTMLParser):
    """Collect href and src targets from a Medetec gallery page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        wanted = {"a": "href", "img": "src", "frame": "src"}.get(tag)
        if wanted is None:
            return
        for name, value in attrs:
            if name == wanted and value:
                self.links.append(value)


def download_medetec() -> None:
    """Medetec wound galleries, at full resolution.

    Site layout:

        files/medetec-image-databases.html          index
        slide scans/<category>/index.html           gallery
        slide scans/<category>/thumbnails/<n>.jpg   ~6 KB preview
        slide scans/<category>/images/<n>.jpg       ~20 KB full size

    Two deliberate narrowings versus a blind recursive mirror:

    1. Only `slide scans/` is walked. `slide scans dressings/` holds product
       photographs of dressing materials, not wounds - a general crawler spends
       most of its time there for nothing.
    2. Full-size images are derived from thumbnail names rather than by
       following each `targetN.html` viewer page. Same result, one request per
       image instead of two, and it avoids a ~600-page breadth-first walk.

    Thumbnails are a fallback only. They are heavily JPEG-quantised, which
    destroys the fine colour and texture detail the Phase 5 features measure.
    """
    out = RAW_DIR / "medetec"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[medetec] -> {out}")

    try:
        index_html = _fetch(MEDETEC_INDEX).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"error: cannot reach Medetec index: {exc}")

    parser = _LinkParser()
    parser.feed(index_html)

    galleries: list[str] = []
    for link in parser.links:
        target = urllib.parse.urljoin(MEDETEC_INDEX, link)
        path = urllib.parse.urlparse(target).path
        if "/slide scans/" in path and path.endswith((".html", ".htm")):
            if target not in galleries:
                galleries.append(target)
    print(f"[medetec] {len(galleries)} wound galleries")

    saved = full_size = fallback = 0
    for gallery in galleries:
        try:
            html = _fetch(gallery).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip gallery {gallery}: {exc}")
            continue
        time.sleep(MEDETEC_DELAY_S)

        gallery_parser = _LinkParser()
        gallery_parser.feed(html)

        for link in gallery_parser.links:
            if "thumbnails/" not in link:
                continue
            thumb_url = urllib.parse.urljoin(gallery, link)
            thumb_path = urllib.parse.urlparse(thumb_url).path
            if Path(thumb_path).suffix.lower() not in IMAGE_SUFFIXES:
                continue

            full_url = thumb_url.replace("/thumbnails/", "/images/")
            rel = urllib.parse.unquote(
                urllib.parse.urlparse(full_url).path.lstrip("/")
            )
            dest = out / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)

            for url, is_full in ((full_url, True), (thumb_url, False)):
                try:
                    dest.write_bytes(_fetch(url))
                except Exception:  # noqa: BLE001 - try the thumbnail next
                    continue
                saved += 1
                full_size += is_full
                fallback += not is_full
                break
            else:
                print(f"  ! no image for {rel}")
            time.sleep(MEDETEC_DELAY_S)

        print(f"  {Path(urllib.parse.urlparse(gallery).path).parent.name}: {saved} total")

    print(f"[medetec] {saved} images ({full_size} full size, {fallback} thumbnail fallback)")


DOWNLOADERS = {
    "cubital": download_cubital,
    "azh": download_azh,
    "medetec": download_medetec,
}


def main() -> None:
    global _SSL_CTX
    _SSL_CTX = _ssl_context()

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
