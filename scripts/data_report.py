#!/usr/bin/env python3
"""Generate docs/data_report.md from the processed datasets.

    python scripts/data_report.py

Written as a script rather than only a notebook so it can run in CI and be
diffed in review; notebooks/data_report.ipynb is a thin wrapper that calls
into this module for interactive exploration.

Outputs:
    docs/data_report.md              summary tables + limitations
    docs/figures/*.png               class distributions, sample grids
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display in CI
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _resolution_stats(paths: list[Path], sample_size: int = 100) -> dict[str, float]:
    """Mean/min/max resolution over a sample - reading every file is wasteful."""
    sample = paths[:: max(1, len(paths) // sample_size)][:sample_size]
    dims = []
    for path in sample:
        img = cv2.imread(str(path))
        if img is not None:
            dims.append(img.shape[:2])
    if not dims:
        return {}
    arr = np.array(dims)
    return {
        "n_sampled": len(dims),
        "mean_h": float(arr[:, 0].mean()),
        "mean_w": float(arr[:, 1].mean()),
        "min_h": int(arr[:, 0].min()),
        "min_w": int(arr[:, 1].min()),
        "max_h": int(arr[:, 0].max()),
        "max_w": int(arr[:, 1].max()),
    }


def _bar_chart(counts: dict[str, int], title: str, out_path: Path) -> None:
    if not counts:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(counts)
    ax.bar(names, [counts[n] for n in names], color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel("images")
    ax.tick_params(axis="x", rotation=45)
    for i, name in enumerate(names):
        ax.text(i, counts[name], str(counts[name]), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _sample_grid(paths: list[Path], title: str, out_path: Path, n: int = 8) -> None:
    if not paths:
        return
    chosen = paths[:: max(1, len(paths) // n)][:n]
    if not chosen:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(4, len(chosen))
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, chosen):
        img = cv2.imread(str(path))
        if img is None:
            continue
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(path.name[:22], fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def report_vein(processed_dir: Path, figures_dir: Path) -> list[str]:
    lines = ["## Vein module (pre-cannulation)", ""]

    manifest_path = processed_dir / "split_manifest.json"
    if not manifest_path.exists():
        lines += [
            "> **Not yet prepared.** CUBITAL could not be downloaded — see the",
            "> blocker note in `docs/dataset_sources.md`. Run",
            "> `python vein_module/src/data_loader.py --prepare` once the raw data",
            "> is in place and re-run this report.",
            "",
        ]
        return lines

    manifest = json.loads(manifest_path.read_text())
    counts = manifest["counts"]

    lines += [
        f"Split level: **{manifest['split_level']}** (seed {manifest['seed']}, "
        f"image size {manifest['image_size']}×{manifest['image_size']})",
        "",
        "| Split | Images | Subjects |",
        "|---|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {counts.get(split, 0)} | {manifest['n_subjects'].get(split, 0)} |"
        )
    lines += [f"| **total** | **{sum(counts.values())}** | "
              f"**{sum(manifest['n_subjects'].values())}** |", ""]

    all_images = _images_in(processed_dir / "train" / "images")
    stats = _resolution_stats(all_images)
    if stats:
        lines += [
            f"Resolution after processing: {stats['mean_h']:.0f}×{stats['mean_w']:.0f} "
            f"(uniform by construction).",
            "",
        ]

    # Vein pixel prevalence drives the class imbalance the Dice loss must absorb.
    masks = _images_in(processed_dir / "train" / "masks")[:100]
    if masks:
        fracs = []
        for path in masks:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                fracs.append((mask > 127).mean())
        if fracs:
            lines += [
                f"Vein pixels occupy **{np.mean(fracs) * 100:.1f}%** of a frame on "
                f"average (n={len(fracs)} sampled). This imbalance is why the Phase 3 "
                "loss pairs BCE with Dice rather than using BCE alone.",
                "",
            ]

    _bar_chart(
        {s: counts.get(s, 0) for s in ("train", "val", "test")},
        "Vein dataset — images per split",
        figures_dir / "vein_split_counts.png",
    )
    _sample_grid(all_images, "Vein — training samples", figures_dir / "vein_samples.png")
    lines += [
        "![vein splits](figures/vein_split_counts.png)",
        "",
        "![vein samples](figures/vein_samples.png)",
        "",
    ]
    return lines


def report_infection(processed_dir: Path, figures_dir: Path) -> list[str]:
    lines = ["## Infection module (post-cannulation)", ""]

    manifest_path = processed_dir / "split_manifest.json"
    if not manifest_path.exists():
        lines += [
            "> **Not yet prepared.** Run",
            "> `python infection_module/src/data_loader.py --prepare`.",
            "",
        ]
        return lines

    manifest = json.loads(manifest_path.read_text())
    counts = manifest["counts"]

    lines += [
        "> **Proxy labels.** " + manifest["label_provenance"],
        "",
        "> **Split caveat.** " + manifest["split_level_caveat"],
        "",
    ]

    classes = sorted({c for split in ("train", "val", "test") for c in counts.get(split, {})})
    lines += ["| Class | Train | Val | Test | Total |", "|---|---:|---:|---:|---:|"]
    for cls in classes:
        row = [counts.get(s, {}).get(cls, 0) for s in ("train", "val", "test")]
        lines.append(f"| {cls} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    totals = [sum(counts.get(s, {}).values()) for s in ("train", "val", "test")]
    lines += [f"| **total** | **{totals[0]}** | **{totals[1]}** | **{totals[2]}** | "
              f"**{sum(totals)}** |", ""]

    proxy = manifest.get("proxy_binary_totals", {})
    if proxy:
        lines += ["Proxy binary target (`is_infected`):", "",
                  "| Split | Wound (1) | Normal skin (0) | Ratio |", "|---|---:|---:|---:|"]
        for split in ("train", "val", "test"):
            p = proxy.get(split, {})
            pos, neg = p.get("positive_wound", 0), p.get("negative_skin", 0)
            ratio = f"{pos / neg:.1f}:1" if neg else "n/a"
            lines.append(f"| {split} | {pos} | {neg} | {ratio} |")
        lines.append("")
        lines += [
            "The positive class outnumbers the negative several times over. Phase 5 "
            "must weight the LightGBM classes or report balanced metrics; raw "
            "accuracy on this split would be misleading.",
            "",
        ]

    external = counts.get("medetec_external_test", {})
    if external:
        lines += [
            f"### External test set — Medetec ({sum(external.values())} images)",
            "",
            "Held out entirely. Different cameras, clinicians, and decade, so "
            "performance here is the honest transfer estimate.",
            "",
            "| Category | Images |",
            "|---|---:|",
        ]
        for cat, n in sorted(external.items(), key=lambda kv: -kv[1]):
            marker = "  ← IV-infiltration injury" if "extravasat" in cat else ""
            lines.append(f"| {cat} | {n}{marker} |")
        lines.append("")

    train_images = _images_in(processed_dir / "train")
    stats = _resolution_stats(train_images)
    if stats:
        lines += [
            f"Native resolution varies: {stats['min_w']}×{stats['min_h']} to "
            f"{stats['max_w']}×{stats['max_h']}, mean "
            f"{stats['mean_w']:.0f}×{stats['mean_h']:.0f} "
            f"(n={stats['n_sampled']} sampled). Resizing happens in the Dataset, "
            "not on disk, so the originals stay available for the ROI crops Phase 5 needs.",
            "",
        ]

    class_totals = {
        cls: sum(counts.get(s, {}).get(cls, 0) for s in ("train", "val", "test"))
        for cls in classes
    }
    _bar_chart(
        class_totals,
        "Infection dataset — images per class (AZH)",
        figures_dir / "infection_class_counts.png",
    )
    if external:
        _bar_chart(
            dict(sorted(external.items(), key=lambda kv: -kv[1])[:12]),
            "Medetec external test — top categories",
            figures_dir / "medetec_categories.png",
        )
    _sample_grid(
        train_images, "Infection — training samples", figures_dir / "infection_samples.png"
    )
    lines += [
        "![class counts](figures/infection_class_counts.png)",
        "",
        "![infection samples](figures/infection_samples.png)",
        "",
    ]
    if external:
        lines += ["![medetec](figures/medetec_categories.png)", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data/processed"))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs/data_report.md"))
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    out_path = Path(args.out)
    figures_dir = out_path.parent / "figures"

    lines = [
        "# Data Report",
        "",
        "Generated by `scripts/data_report.py`. Do not edit by hand.",
        "",
    ]
    lines += report_vein(processed / "vein", figures_dir)
    lines += report_infection(processed / "infection", figures_dir)
    lines += [
        "## What this data cannot tell us",
        "",
        "1. **No infection ground truth.** Nothing here labels a wound infected. "
        "The binary target is a wound-vs-skin proxy.",
        "2. **No longitudinal series.** No dataset photographs the same site on "
        "consecutive days, so the Phase 5 temporal tracker has no real validation "
        "set — only the synthetic progression notebook.",
        "3. **No cannula images.** Phase 4 needs ~100–200 hand-labelled hub images "
        "that do not exist publicly.",
        "4. **No IV sites at all.** Every wound image here is a chronic or surgical "
        "wound, not a cannula insertion site. The extravasation gallery is the "
        "nearest analogue and it is small.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print(f"figures -> {figures_dir}")


if __name__ == "__main__":
    main()
