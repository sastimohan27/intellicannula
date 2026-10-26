# Review 2 — Detection Correctness: What Was Wrong, What Changed, Measured Results

All numbers below were produced by `vein_module/src/evaluate.py` on the
held-out **test split (302 images, 288 with an annotated vein; subjects
disjoint from training)**. The before/after rows were run with the same code
and the same images.

## 1. Problems found (audit)

| # | Symptom | Root cause |
|---|---|---|
| 1 | Vein mask was thick blobs + speckle; ~74% of highlighted pixels were not vein | Model stopped at **epoch 6 of 25** (val Dice still rising); fixed 0.5 threshold; no speckle removal |
| 2 | A clear 13.8 px vein graded **"Not Recommended"** | Grade used `confidence`, which is the *margin between the best and runner-up candidate* — several good veins → low margin → bad grade |
| 3 | A **landscape photo** got infection risk **0.999** | When no cannula was detected (or the detector crashed), the code silently scored the image centre |
| 4 | "Cannot reach the API" on vein upload | API ran in a venv without OpenCV; the crash (500) had no CORS headers, so the browser reported a network error |
| 5 | Chart tooltip showed `index` and an unnamed value | A `Scatter` series inside the `LineChart` added its own tooltip rows |

## 2. Fixes

**Vein segmentation**
- Resumed training (added `--resume` to `train.py`) from epoch 6 → best checkpoint **epoch 16** (val Dice 0.353 → 0.475; epochs 17–19 did not improve as the cosine learning rate reached ~0).
- Post-processing in one shared module, `vein_module/src/postprocess.py`, used by **both** the API and the evaluator, so reported metrics are measured on exactly the mask the API serves:
  probability threshold **0.9**, drop connected blobs **< 60 px**.
- Both values were tuned on the **validation** split, never on test. 150 px scored marginally higher on val (0.601 vs 0.590) but deletes short genuine veins, so 60 px was kept.
- Checked where the false positives are: with the retrained model, **0% of false-positive pixels fall off the arm** (measured against CUBITAL's arm label). The bedsheet false positives in the earlier report are gone; the remaining errors are vein margins and faint unlabelled veins.

**Vein grade** — now based on vein width only; "Not Recommended" only when no site exists. Confidence is shown separately with its reason (e.g. "6 candidates scored similarly").

**Cannula — strict gate**
- A site check is scored **only if the detector finds a hub at confidence ≥ 0.5**. Otherwise the API returns **422 "No cannula detected… retake"** and nothing is stored.
- No centre-of-image fallback. Missing or crashing detector → **503**, never a guess.
- The threshold of 0.5 was chosen from measurements on images with no cannula (below).

**Engineering** — missing ML dependencies return 503 with the fix instead of a fake network error; tooltip fixed; regression tests added (API 39/39, modules 188/188 pass).

## 3. Results

### Vein segmentation (test split, 288 images with a vein)

| Metric | Before (epoch 6, thr 0.5) | Epoch 13 | **Final (epoch 16, thr 0.9, min 60 px)** | Change |
|---|---:|---:|---:|---:|
| **Dice** | 0.370 | 0.557 | **0.567** | +53% |
| **IoU** | 0.239 | 0.403 | **0.413** | +73% |
| **Precision** | 0.262 | 0.482 | **0.495** | +89% |
| Recall | 0.814 | 0.751 | 0.758 | −0.06 |

Reading it: precision nearly doubled — the system now highlights far fewer
non-vein pixels. Recall dropped slightly: the stricter threshold misses some
faint vein edges. That trade is deliberate for a tool whose output a
clinician looks at.

### Non-vein images (no hallucinated veins)

Uploaded to the vein-scan endpoint: a landscape photo, a real IV-hand photo, a
wound photo → **no site returned for any of them**. Landscape and wound:
"empty mask — no veins segmented". IV hand: a few small fragments, rejected
because "no segment met the minimum bar".

### Cannula detector — images with **no** cannula (top confidence)

| Image set | n | ≥ 0.4 (old threshold) | ≥ 0.5 (new) | Max |
|---|---:|---:|---:|---:|
| CUBITAL forearm NIR | 40 | 2 | **0** | 0.474 |
| Medetec wound photos | 40 | 0 | 0 | 0.157 |
| AZH wound photos | 40 | 0 | 0 | 0.098 |
| Landscape photo | 1 | 0 | 0 | 0.236 |

Real IV photos: dressing photo **detected (0.79)**; hand photo **missed (0.19)** → rejected with "retake", not scored.

## 3b. Infection monitoring — redesigned

**Problem found:** every site check scored ~1.00, including a clean IV-hand
photo (0.999). The LightGBM model was trained on AZH close-up *chronic wounds vs
normal skin*, so any IV photo — tape, plastic hub, dressing — looked "not normal
skin". It answered "is this a wound?", not "is this site reddening?". The
"VIP score" was that number relabelled; real VIP also needs pain, swelling and a
palpable cord, which a photo cannot show.

**Replacement** (`infection_module/src/erythema.py`): measure the visible sign of
phlebitis directly — redness spreading from the insertion point.

1. Photo-quality gate: blurred, too dark or overexposed → retake.
2. Cannula gate (≥ 0.5) → hub box.
3. Ring of skin 0.75–5 hub half-sizes around the hub; YCrCb skin mask removes
   tape, dressing, plastic and background.
4. Reference = median a\* of the farthest 30% of visible skin in the same photo
   (cancels lighting and skin tone).
5. Reddened = ring skin with Δa\* > 6 (CIELAB units; above the ~2–3 unit
   just-noticeable difference). Score = reddened share, saturating at 50%.
6. The day-0 photo is measured the same way; change since baseline is a
   difference of two measurements, so no fragile image warping is needed.

**Functional tests** (`infection_module/tests/test_erythema.py`, 7 pass): clean
site reads 0; score rises monotonically with added redness; a uniform tone shift
is not redness; tape and hub plastic are excluded; a site with no visible skin
is refused.

**One bug the functional test caught:** with the reference band at 2.5–3.5
hub-sizes, spreading redness reddened the reference too and the score *fell* as
redness was added. The reference now uses the farthest skin in the photo.

**Simulated 3-day sequence** (drawn hub on a real limb photo; redness added —
files are named `SIMULATED_*`):

| Day | Reddened skin | Change | Score | Level | Alert |
|---|---:|---:|---:|---|---|
| 1 (baseline) | 5% | — | 0.11 | No visible redness | no |
| 2 | 12% | +7 pts | 0.25 | Mild | no |
| 3 | 58% | +52 pts | 1.00 | Extensive | **yes** |

This shows the pipeline responds correctly. It is **not** evidence that it
detects real phlebitis — that needs nurse-scored photos of real IV sites.

## 4. Limitations (stated honestly)

- **Cannula detector is trained on 90 synthetic images** (procedurally drawn hubs on real skin). Its 0.995 mAP is on synthetic data and does not transfer; it misses some real cannulas. Real labelled photos are the next step. The gate means a miss produces a retake request, not a wrong score.
- Vein model plateaued at epoch 16 (val Dice flat through epoch 19). Further gains need more or better data, not more epochs. The run can still be continued with `python -m vein_module.src.train --epochs 25 --patience 8 --resume vein_module/models/resume_from.pt`. Re-tune the threshold afterwards.
- CUBITAL is a **paediatric NIR** cohort (age 5–22). Ordinary RGB photos do not show subsurface veins the same way; results on RGB are not claimed.
- "Empty masks correctly left blank" on the 14 vein-free test images is 0% (epoch 16; 7% at epoch 13) — the model still marks something on most images with no labelled vein.
- Redness thresholds (Δa\* > 6, levels at 10/25/50%) are documented heuristics,
  not clinically calibrated. A close-up photo with no normal skin away from the
  site is refused; diffuse redness filling the whole frame reads low against
  itself, which is why the day-0 baseline matters.
- Needle angle is a heuristic proxy, not a depth measurement. All outputs are advisory decision support, not clinical recommendations.

## 5. Demo script

Demo images: `~/Desktop/IntelliCannula_demo/`. Use a **new patient** (the first
site photo becomes the baseline).

1. Open `http://localhost:3000` (or `http://<laptop-IP>:3000` on a phone on the
   same Wi-Fi — the upload buttons open the camera), create a patient.
2. **Vein scan** → `1_vein_scan_arm.png`: thin vein lines, site on a vein,
   grade + confidence with reason.
3. **Site check** → `SIMULATED_day1_baseline.png`: "No visible redness",
   overlay shows hub (green), measured ring (cyan), counted skin (red).
4. **Site check** → `SIMULATED_day2_mild.png`, then `SIMULATED_day3_spreading.png`:
   redness rises, alert fires on day 3, chart climbs.
5. **Site check** → `3_site_check_landscape_REJECTED.png`: "No cannula detected… retake".

Reproduce the numbers:

```bash
infection_module/.venv/bin/python -m vein_module.src.evaluate --threshold 0.9 --min-component-px 60
cd backend && ../infection_module/.venv/bin/python -m pytest -q
```
