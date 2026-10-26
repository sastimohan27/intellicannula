"""Run the faculty-review demo end to end against the running API, and check it.

    infection_module/.venv/bin/python scripts/run_demo.py [--api http://localhost:8001]

Creates a fresh DEMO- patient (demo records stay separate from real ones),
uploads every file in data/demo/ to the workflow its modality supports, and
asserts the expected outcome. Exits non-zero if any step deviates. Build the
images first with scripts/build_demo_set.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

DEMO = Path(__file__).resolve().parents[1] / "data/demo"

# (file, endpoint, expected HTTP status, check on the JSON body, description)
STEPS = [
    ("1_vein_scan_arm.png", "vein-scan", 201,
     lambda b: b["modality_status"] == "verified" and b["acquisition_source"] == "cubital_dataset",
     "NIR frame accepted and segmented"),
    ("4_rgb_photo_for_vein_REJECTED.png", "vein-scan", 422,
     lambda b: "NIR" in b["detail"], "RGB photo refused for vein scan"),
    ("SIMULATED_day1_baseline.png", "site-check", 201,
     lambda b: b["assessment_status"] == "baseline_recorded" and b["sample_origin"] == "simulated_demo",
     "baseline recorded, labelled simulated"),
    ("SIMULATED_day2_mild.png", "site-check", 201,
     lambda b: b["comparison_status"] == "compared" and not b["alert_triggered"],
     "compared with baseline, no alert"),
    ("SIMULATED_day3_spreading.png", "site-check", 201,
     lambda b: b["assessment_status"] == "review_recommended" and b["alert_triggered"],
     "change detected, alert raised"),
    ("3_site_check_landscape_REJECTED.png", "site-check", 422,
     lambda b: "cannula" in b["detail"].lower(), "no-cannula photo refused"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8001")
    args = parser.parse_args()

    missing = [name for name, *_ in STEPS if not (DEMO / name).exists()]
    if missing:
        print(f"missing demo images {missing}; run scripts/build_demo_set.py first")
        return 2

    client = httpx.Client(base_url=args.api, timeout=120)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    patient = client.post("/patients", json={"name": "Demo Patient (synthetic)", "mrn": f"DEMO-{stamp}"}).json()
    print(f"demo patient #{patient['id']} ({patient['mrn']})\n")

    failures = 0
    for name, endpoint, status, check, description in STEPS:
        with (DEMO / name).open("rb") as handle:
            start = time.perf_counter()
            response = client.post(f"/patients/{patient['id']}/{endpoint}",
                                   files={"file": (name, handle, "image/png")})
            elapsed = (time.perf_counter() - start) * 1000
        body = response.json()
        ok = response.status_code == status and check(body)
        failures += not ok
        detail = (body.get("detail") or body.get("assessment_label") or
                  f"site {body.get('insertion_x')},{body.get('insertion_y')} grade {body.get('vein_grade')}")
        extra = ""
        if endpoint == "site-check" and response.status_code == 201:
            extra = (f" | reddened {body['reddened_fraction']:.0%}"
                     + (f", change {body['change_since_baseline']:+.0%}" if body["change_since_baseline"] is not None else ""))
        print(f"{'PASS' if ok else 'FAIL'}  {name:38s} {endpoint:10s} {response.status_code}  "
              f"{elapsed:6.0f} ms  {description}: {str(detail)[:70]}{extra}")

    history = client.get(f"/patients/{patient['id']}/history").json()
    print(f"\ntimeline: {history['n_checkins']} check-ins (rejected photos create none), "
          f"scores {[round(p['risk_score'], 2) for p in history['points']]}")
    if history["n_checkins"] != 3:
        failures += 1
        print("FAIL  expected exactly 3 stored check-ins")
    print(f"\n{'all steps passed' if not failures else f'{failures} step(s) failed'} · "
          f"open http://localhost:3000/patients/{patient['id']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
