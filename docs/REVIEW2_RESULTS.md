# IntelliCannula — Final Review: Changes, Measured Results, Limitations

Every number here was measured in this repository; the command that produced
it is given. Historical baselines are kept so before/after is visible.
Detail: [`vein_error_analysis.md`](vein_error_analysis.md) ·
raw data: [`vein_postprocess_experiments.json`](vein_postprocess_experiments.json),
[`vein_error_analysis.json`](vein_error_analysis.json),
[`cannula_gate_eval.json`](cannula_gate_eval.json).

## 1. Problems found and fixed

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Vein mask was blobs + speckle; 74% of marked pixels not vein | Training stopped at epoch 6/25; fixed 0.5 threshold | Resumed to epoch 16; threshold 0.9 + 60 px speckle filter tuned on val |
| 2 | Clear 13.8 px vein graded "Not Recommended" | Grade used the selector's *separation* margin | Grade from vein width; refusal → "Not Recommended" |
| 3 | Landscape photo scored infection risk 0.999 | No cannula → silently scored image centre | Strict cannula gate; no fallback region |
| 4 | Every site photo scored ~1.00 | LightGBM trained on *chronic wound vs normal skin* (proxy labels) | Replaced by measured peri-cannula redness vs patient's baseline |
| 5 | Any image accepted for vein scanning, incl. RGB phone photos | No modality check | **NIR provenance gate** (§3) — RGB never reaches the model |
| 6 | Follow-up of a different scene compared as if same site | No scene check | Landmark (ORB/RANSAC) match against baseline; < 50 → refused |
| 7 | Site check took 1.7–2.8 s per detection | Detector ran in a fresh subprocess per request | In-process, cached model: 42 ms |
| 8 | Vein model on CPU, 597 ms forward pass | Default device | MPS: 77 ms, identical masks (40/40 test images, Dice 1.0000) |
| 9 | Chart times 5.5 h early (IST) | API emitted UTC without "Z" | UTC normalisation; regression tests |
| 10 | "+0.89 / day" trend | Per-check-in change labelled per day | "+0.89 since last check-in" |
| 11 | Fake "VIP score" | Risk number relabelled | Removed; VIP needs signs a photo cannot show |
| 12 | Dashboard claimed Dice 0.37, LightGBM F1 0.988, "mAP 0.995 (Trained)" | Stale hard-coded text | Live versions from `/models` + measured results with caveats |
| 13 | Change since baseline under-measured (r 0.59 vs injected truth) | Each photo referenced to its own far skin, which spreading redness also reddens | Pixel-wise change vs aligned baseline, colour-matched on neutrals away from the site (r 0.995) |
| 14 | Most-inflamed follow-ups refused (13/48) | Detector confidence drops on red skin | Weak detection accepted on follow-ups matching a verified baseline |
| 15 | False "change" from white balance when nothing white is in frame | No colour reference | Reported as "unable to compare reliably"; capture guidance asks for a white sheet/paper |

## 2. Vein segmentation

**Experiments (validation split only, 285 images with a vein, epoch-16 checkpoint):**

| Configuration | Dice | Precision | Recall | |
|---|---:|---:|---:|---|
| **thr 0.90, min 60 px (deployed)** | **0.575** | 0.501 | 0.759 | kept |
| thr 0.95 | 0.592 | 0.557 | 0.701 | higher Dice, but 6 pts more vein missed — not adopted |
| thr 0.85 | 0.562 | 0.473 | 0.788 | worse |
| flip TTA, thr 0.90 | 0.580 | 0.515 | 0.748 | +0.005 for 2× latency — not adopted |
| hysteresis 0.9/0.5 · 0.9/0.7 | 0.517 · 0.540 | 0.399 · 0.434 | 0.856 · 0.826 | worse |

**No inference-time change was established as an improvement worth adopting**,
so the deployed configuration and its single test evaluation are unchanged.

**Test split (288 images with a vein; subjects disjoint from training):**

| Metric | Original (epoch 6, thr 0.5) | Final (epoch 16, thr 0.9, min 60 px) |
|---|---:|---:|
| Dice | 0.370 | **0.567** (median 0.610, P10 0.350, P90 0.741) |
| IoU | 0.239 | **0.413** |
| Precision | 0.262 | **0.495** |
| Recall | 0.814 | 0.758 |

**Errors:** 47% of false-positive pixels are within 2 px of a true vein (drawn
too wide), 53% on the arm away from annotated veins, 0.3% off the arm. Thin-vein
recall 0.70 vs 0.76 for wider veins. Veins are often fragmented (median 1.5
predicted pieces per annotated vein). High-contrast frames score *lowest* (0.536
vs 0.591 for low contrast), and the worst case shows a clear vein the annotation
omits — some "false positives" are likely unannotated veins (hypothesis; needs
re-annotation to measure).

**Integrity:** 0 subjects and 0 identical images shared between splits; 0
mask-pairing errors (veins outside the annotated arm).

**Latency (Apple MPS):** model forward 77 ms; vein pipeline in the API 88–127 ms
warm; whole request 185 ms warm, 3.4 s on the first request (model load +
NIR-registry build).

Commands: `python -m vein_module.src.postprocess_experiments` (val),
`python -m vein_module.src.error_analysis` (test, analysis only).

## 3. NIR-only vein scanning

Pixel statistics **cannot** establish NIR here: CUBITAL's NIR camera renders a
purple tint, and its colour spread (CIELAB chroma SD median 10.2) overlaps RGB
photos (5th pct 2.8, median 7.5). So acceptance is by **provenance**
(`backend/app/core/modality.py`):

| Input | Result |
|---|---|
| SHA-256 matches a local CUBITAL NIR frame | **verified** — `cubital_dataset`, with its split (train/val/test) |
| `X-NIR-Device` + matching `X-NIR-Device-Token` (server-side registry, env `NIR_DEVICES`) | **verified** — `registered_device` |
| File has camera EXIF (Make/Model), e.g. a phone photo | **rejected** — "not accepted for vein segmentation… RGB camera" |
| Anything else (grayscale, "NIR" filename, device ID without token…) | **rejected** — "NIR compatibility unverified" |

The model is never called on a rejection (asserted in `tests/test_modality.py`).
No live NIR device is attached to this prototype, so only verified dataset
frames are accepted today; the UI states this (`/modality/nir`). The vein upload
does not open the phone camera — a phone camera is RGB.

## 4. Cannula gate (`docs/cannula_gate_eval.json`)

States: ≥ 0.50 verified · 0.25–0.50 "unable to verify" (rejected, says so) · < 0.25 no cannula.

| Data | n | False acceptance at 0.50 | False rejection at 0.50 |
|---|---:|---:|---:|
| Synthetic val (threshold comparison only) | 10 pos / 5 neg | 0% (0.3–0.4: 20%) | 0% (0.7: 30%) |
| Synthetic test | 19 / 6 | 0% | 5% (1/19) |
| Real, no cannula — CUBITAL forearms | 200 | **1%** (+19% "unable to verify") | — |
| Real, no cannula — AZH wounds | 100 | **1%** | — |
| Real, no cannula — Medetec wounds | 100 | **2%** | — |
| Real IV photos | 2 | — | dressing 0.79 accepted; hand 0.19 **missed** |

This corrects the earlier "0 of 121" figure, which came from a smaller sample.
A false acceptance on a *follow-up* is still stopped by the scene check (§5);
on a *baseline* photo it is not. No labelled real cannula set exists, so
real-world false rejection is unmeasured beyond the photos listed. AZH/Medetec
images were also used as backgrounds for the synthetic training set.

## 5. Site-change monitoring

**Controlled validation:** [`site_monitoring_validation.md`](site_monitoring_validation.md).
On real limb photos with known injected redness, the adopted baseline-anchored
change measurement agrees with the truth at r 0.995–0.996 (mean error 3.0–3.6
points), detects every visible change, and raised 0 false alarms on 27
lighting / white-balance / JPEG / phone-movement photos. It replaced the
original measurement (r 0.59, mean error 21.5 points). Only 4 of 12 backgrounds
were usable (detector misses), and comparisons with no white/grey reference in
frame are deferred, not guessed.

Pipeline: quality gate (resolution, blur, darkness, overexposure) → cannula gate
→ redness in a skin ring around the hub (`infection_module/src/erythema.py`) →
comparison with the patient's baseline.

**Safeguards:** missing/unusable baseline → "unable to compare"; different scene
(< 50 RANSAC landmark matches) → refused. Measured: same scene incl. moved,
rotated 8° and re-saved JPEG 881–1,865 matches; unrelated scenes ≤ 11. Real
day-to-day photos will vary more than this simulated set — the threshold needs
checking on real follow-ups. Uniform lighting/white-balance shifts are cancelled
by the in-photo reference (functional test).

**Statuses:** Baseline recorded · No significant visible change · Visible
change detected (≥ +10 pts or level ≥ 2) · Review recommended (alert) · Unable
to compare. There is deliberately no "urgent" tier: that needs a clinically
reviewed escalation protocol.

**Simulated sequence** (procedural hub on a real limb photo, redness added;
each PNG carries an `IntelliCannula-Simulated` marker so renaming cannot hide it):

| Day | Reddened | Change | Score | Status |
|---|---:|---:|---:|---|
| 1 | 5% | — | 0.11 | Baseline recorded |
| 2 | 12% | +7 pts | 0.25 | No significant visible change |
| 3 | 58% | +52 pts | 1.00 | Review recommended (alert) |

This shows the pipeline responds correctly; it is **not** evidence of detecting
real phlebitis.

## 6. Tests (all executed)

| Suite | Command | Result |
|---|---|---|
| API | `cd backend && ../infection_module/.venv/bin/python -m pytest -q` | **68 passed** (39 original + 29 new) |
| ML modules | `infection_module/.venv/bin/python -m pytest -q` | **192 passed** |
| Site-change validation | `cd backend && ../infection_module/.venv/bin/python ../scripts/validate_site_monitoring.py` | r 0.995, 0 false alarms (§5) |
| Frontend unit | `cd frontend && npm test` | **5 passed** |
| Frontend static | `npx tsc --noEmit` · `npx eslint app components lib tests` | clean |
| End-to-end demo | `infection_module/.venv/bin/python scripts/run_demo.py` | **6/6 steps** pass, 3 check-ins stored |

The original 39 API tests are unchanged in their bodies; their mocked-ML
fixtures now also stub the NIR check (documented in `conftest.py`), and
`test_modality.py` exercises the real check. UI verified by driving headless
Chrome through the real upload inputs (results and rejection screens).

## 7. Demo

```bash
infection_module/.venv/bin/python scripts/build_demo_set.py   # -> data/demo/ + manifest.json
infection_module/.venv/bin/python scripts/run_demo.py         # scripted check against the API
```

Manually: dashboard → **Create demo patient** (MRN `DEMO-…`) → vein scan
`1_vein_scan_arm.png` (CUBITAL frame 942, **test split**, median Dice) → site
check `SIMULATED_day1`, `day2`, `day3` → site check
`3_site_check_landscape_REJECTED.png` (no cannula) → vein scan
`4_rgb_photo_for_vein_REJECTED.png` (not NIR).

## 8. Limitations

- Vein segmentation still misses ~24% of annotated vein pixels and half of what
  it marks is outside the annotation; paediatric NIR cohort (ages 5–22); trained
  on one capture rig.
- No NIR capture hardware is attached; phone cameras are RGB and cannot be used
  for vein scans.
- Cannula detector: 90 synthetic training images; 1–2% false acceptance on real
  no-cannula images; misses some real cannulas.
- Redness thresholds and statuses are documented heuristics, not clinically
  validated; redness alone cannot confirm or exclude infection.
- Scene-match threshold validated only on simulated/transformed images.
- Patient records are not authenticated or encrypted — synthetic or
  de-identified data only. No patient-deletion endpoint exists; records persist
  in the SQLite/Postgres database until removed by an administrator.
- Check-ins stored before this release (patients created before 2026-10-21)
  were scored by the retired LightGBM model and read ~1.00; use new patients.
- Needle angle is a heuristic, not a depth measurement.

## 9. Future work

1. NIR capture device (850 nm illumination + IR-pass camera) registered via
   `NIR_DEVICES`, and images from it for fine-tuning and evaluation.
2. Ethics-approved collection of real IV-site photos with daily nurse VIP
   scores, to calibrate redness thresholds and validate the scene check.
3. 300–1,000 labelled real cannula photos to retrain and measure the detector.
4. Re-annotation of a sample of "false-positive" vein regions.
5. Authentication, audit review UI and encrypted storage before any real data.
