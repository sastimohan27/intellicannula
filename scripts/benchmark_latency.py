#!/usr/bin/env python3
"""End-to-end latency benchmark for both ML pipelines.

    python scripts/benchmark_latency.py --runs 20

Times image-in to result-out, which is what a clinician actually waits for -
not just the forward pass. Reports mean and p95; p95 matters more than mean
here, because the occasional slow request is the one that gets noticed at the
bedside.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile - no interpolation, valid for small n."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1)
    return ordered[index]


def _summarise(name: str, timings: list[float], device: str) -> dict:
    return {
        "pipeline": name,
        "device": device,
        "runs": len(timings),
        "mean_ms": round(statistics.mean(timings), 1) if timings else 0.0,
        "median_ms": round(statistics.median(timings), 1) if timings else 0.0,
        "p95_ms": round(_percentile(timings, 95), 1),
        "min_ms": round(min(timings), 1) if timings else 0.0,
        "max_ms": round(max(timings), 1) if timings else 0.0,
    }


def benchmark_vein(image_path: Path, runs: int, device: str) -> dict | None:
    from app.core.ml import run_vein_scan

    data = image_path.read_bytes()
    run_vein_scan(data)  # warm-up: excluded, it loads the checkpoint

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        run_vein_scan(data)
        timings.append((time.perf_counter() - started) * 1000)
    return _summarise("vein-scan", timings, device)


def benchmark_site_check(image_path: Path, runs: int, device: str) -> dict | None:
    from app.core.ml import run_site_check

    data = image_path.read_bytes()
    run_site_check(data, 1, None)  # warm-up

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        run_site_check(data, 1, data)  # with a baseline, so registration runs
        timings.append((time.perf_counter() - started) * 1000)
    return _summarise("site-check", timings, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "docs/latency_benchmark.json")
    )
    args = parser.parse_args()

    vein_images = sorted((REPO_ROOT / "data/processed/vein/test/images").glob("*.png"))
    site_images = sorted(
        (REPO_ROOT / "data/processed/infection/test/normal").glob("*.jpg")
    )

    results = []
    if vein_images:
        print(f"benchmarking vein-scan ({args.runs} runs)...")
        results.append(benchmark_vein(vein_images[0], args.runs, args.device))
    else:
        print("no processed vein images - skipping vein-scan")

    if site_images:
        print(f"benchmarking site-check ({args.runs} runs)...")
        results.append(benchmark_site_check(site_images[0], args.runs, args.device))
    else:
        print("no processed infection images - skipping site-check")

    for r in results:
        print(
            f"  {r['pipeline']:<12} {r['device']:<4} "
            f"mean {r['mean_ms']:>8.1f} ms  p95 {r['p95_ms']:>8.1f} ms  "
            f"(min {r['min_ms']:.0f} / max {r['max_ms']:.0f})"
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
