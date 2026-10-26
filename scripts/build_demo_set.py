"""Build the faculty-review demo set in data/demo/ (git-ignored), deterministically.

    infection_module/.venv/bin/python scripts/build_demo_set.py

Every file is routed to the workflow its modality supports, and the manifest
says so:

| file | modality | workflow | expected |
|---|---|---|---|
| 1_vein_scan_arm.png | NIR (CUBITAL frame, verified by SHA-256) | vein scan | segmented |
| SIMULATED_day1..3 | RGB, simulated hub + redness | site check | baseline -> change -> alert |
| 3_site_check_landscape_REJECTED.png | RGB, no cannula | site check | rejected |
| 4_rgb_photo_for_vein_REJECTED.png | RGB photo | vein scan | rejected (not NIR) |

Sources are local files that are not committed (dataset licences / copyright):
the CUBITAL NIR frames, and two photos uploaded during development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "backend/data/uploads/patients/1/site"
SOURCES = {
    "hand": UPLOADS / "20261009/ac619862d2ffa641.png",       # real IV-hand photo (RGB)
    "landscape": UPLOADS / "20261014/1f616f6a726f9413.png",  # lake landscape (RGB)
}
# Held-out TEST-split frame with typical (near-median) Dice for the final model,
# chosen in docs/vein_error_analysis.md - not a training image, not cherry-picked.
DEFAULT_VEIN_FRAME = "942"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data/demo"))
    parser.add_argument("--vein-frame", default=DEFAULT_VEIN_FRAME)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in SOURCES.values() if not p.exists()]
    nir = ROOT / f"data/raw/cubital/square_dataset/nir_images/{args.vein_frame}.png"
    if not nir.exists():
        missing.append(str(nir))
    if missing:
        raise SystemExit("missing demo sources:\n  " + "\n  ".join(missing))

    split = next((s for s in ("train", "val", "test")
                  if (ROOT / f"data/processed/vein/{s}/images/cubital_{args.vein_frame}.png").exists()), None)

    shutil.copyfile(nir, out / "1_vein_scan_arm.png")
    shutil.copyfile(SOURCES["landscape"], out / "3_site_check_landscape_REJECTED.png")
    shutil.copyfile(SOURCES["hand"], out / "4_rgb_photo_for_vein_REJECTED.png")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_demo_sequence.py"), str(SOURCES["hand"]), str(out),
         "--centre", "250", "720", "--width-frac", "0.22"],
        check=True, capture_output=True,
    )

    manifest = {
        "1_vein_scan_arm.png": {"modality": "NIR", "provenance": f"CUBITAL frame {args.vein_frame} ({split} split)",
                                "workflow": "vein-scan", "expected": "accepted; segmented"},
        "SIMULATED_day1_baseline.png": {"modality": "RGB (simulated)", "workflow": "site-check",
                                        "expected": "baseline recorded"},
        "SIMULATED_day2_mild.png": {"modality": "RGB (simulated)", "workflow": "site-check",
                                    "expected": "mild redness"},
        "SIMULATED_day3_spreading.png": {"modality": "RGB (simulated)", "workflow": "site-check",
                                         "expected": "review recommended (alert)"},
        "3_site_check_landscape_REJECTED.png": {"modality": "RGB", "workflow": "site-check",
                                                "expected": "rejected: no cannula"},
        "4_rgb_photo_for_vein_REJECTED.png": {"modality": "RGB", "workflow": "vein-scan",
                                              "expected": "rejected: NIR compatibility unverified"},
    }
    for name in manifest:
        manifest[name]["sha256"] = sha(out / name)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, row in manifest.items():
        print(f"{name:40s} {row['modality']:18s} -> {row['workflow']:10s} {row['expected']}")


if __name__ == "__main__":
    main()
