# Review 1 — Preparation Notes

**IntelliCannula** · J Component · 20 marks
Faculty asked you to be thorough about what you implemented, not to present slides.

Everything below is verified against the repository. If a number here disagrees
with something you remember, trust this file — it was regenerated from the
actual reports.

---

## The 60-second answer

> IntelliCannula is clinical decision support for peripheral IV cannulation, in
> two modules. **Before** insertion, it takes a near-infrared forearm image,
> segments the veins with an Attention U-Net, and recommends an insertion point
> and needle angle. **After** insertion, it takes daily photos of the same IV
> site, registers each against the patient's baseline, and tracks colour and
> texture change to flag developing phlebitis before it becomes visible to the
> eye.
>
> It recommends; it never acts. Every output carries a confidence and a reason,
> and the system refuses when it cannot support a recommendation.

If they ask "why does this matter" — **first-attempt cannulation failure runs
around 30–40% in difficult patients**, and phlebitis is one of the most common
hospital-acquired complications. Both are ordinary, high-volume problems where a
small improvement compounds.

---

## Rubric 1 · Problem Understanding (3 marks)

### The clinical problem, in two halves

| Phase | Problem | What we output |
|---|---|---|
| Pre-cannulation | Finding a vein is hard in obese, elderly, dehydrated, or dark-skinned patients. Repeated sticks cause pain, delay, and vessel damage. | Insertion point (x, y), needle angle, confidence |
| Post-cannulation | Phlebitis and infiltration develop gradually and are judged by eye, inconsistently, by whoever is on shift. | Risk score 0–1, trend, alert with a stated reason |

### The insight that ties them together

Both halves are the *same site at different times*. That is why the system is
patient-anchored rather than image-anchored: the post-cannulation module
compares a site **against that patient's own baseline**, not against a
population average. Skin tone and lighting vary enormously between patients;
change at one site over time is far more informative than absolute redness.

### Be ready to say what you *cannot* claim

This scores better than overclaiming. Three honest limits:

1. **No public dataset labels an IV site as infected.** Our target is a proxy.
2. **No public dataset photographs the same site on consecutive days.** The
   temporal logic is unvalidated against outcomes.
3. **No public dataset of cannula sites exists at all.** Phase 4 is blocked on
   manual data collection.

---

## Rubric 2 · System Design & Methodology (5 marks)

### Architecture

```
        Next.js dashboard  (capture, overlay, risk badge, trend chart)
                 │  REST + WebSocket
        FastAPI backend    (validation, audit log, model versioning)
           ├── vein_module        NIR → Attention U-Net → site selector
           └── infection_module   RGB → registration → features → LightGBM
                 │
        PostgreSQL + image store  (S3, or local disk fallback)
```

### Pre-cannulation pipeline

1. **Preprocess** — CLAHE on the L channel in LAB space, then a bilateral filter.
   Bilateral rather than Gaussian because veins are low-contrast ridges a few
   pixels wide; a Gaussian wide enough to kill sensor noise erases them too.
2. **Build a 3-channel input** — `[preprocessed R, preprocessed G, Frangi
   vesselness]`. Blue is dropped: haemoglobin absorption falls off through the
   visible range, so blue carries mostly surface scatter.
3. **Segment** — Attention U-Net, **7,850,573 parameters** (budget was <15M for
   bedside deployability).
4. **Select a site** — skeletonise the mask, measure local width by distance
   transform, find straight runs with a sliding-window R² > 0.95, score them
   `0.5·width + 0.3·length + 0.2·(1 − edge proximity)`.
5. **Estimate angle** — 15° (superficial) to 30° (deep), interpolated on width
   and contrast.

### Post-cannulation pipeline

1. **Detect the cannula hub** (YOLOv11) to anchor the ROI — *not trained*,
   falls back to a centre-of-frame ROI.
2. **Register** against the patient's baseline — ORB keypoints, ratio test 0.75,
   RANSAC homography. Falls back to translation-only if fewer than 10 good
   matches, and **reports that it did**.
3. **Extract 30 features** — HSV/LAB colour statistics, LAB distance from a
   healthy-skin patch sampled in the same photo, LBP histogram, GLCM over four
   angles.
4. **Score risk** — LightGBM, deliberately small (`num_leaves=15, max_depth=4`)
   given ~600 training rows.
5. **Alert** on three ORed rules: risk > 0.6, jump > 0.15 since last visit, or
   three consecutive increases.

### Design decisions you should be able to defend

**Why Attention U-Net rather than plain U-Net?**
Veins are ~0.45% of a frame. Encoder skip connections carry mostly irrelevant
skin. Attention gates let the coarser decoder stage suppress those regions
before concatenation. Gating is *soft*, so a faint vein the network is unsure
about is dimmed, not deleted.

**Why BCE + Dice rather than either alone?**
Measured: vein pixels are **0.358%** of a frame, a **278:1** imbalance. BCE
alone is minimised by predicting background everywhere. Dice is scale-invariant
to imbalance but has unstable gradients on empty masks. Together they cover each
other.

**Why three alert rules instead of one threshold?**
Rule 1 catches a site that is already bad. Rule 2 catches fast deterioration
while the absolute level is still moderate. Rule 3 catches the slow monotone
creep that never trips either — which is how cellulitis often presents.

**Why a per-image healthy-skin reference?**
So the erythema measure works across skin tones. An absolute redness threshold
would be calibrated for one complexion and fail on others.

---

## Rubric 3 · Implementation Progress (5 marks — needs ≥30%)

**You are at roughly 90% of the plan, far past the 30% bar.** Lead with the
verified end-to-end run, not with the file count.

### What is complete

| Phase | Status |
|---|---|
| 1 Scaffolding | Done |
| 2 Data pipeline | Done, on real data |
| 3 Vein module | Trained on CUBITAL |
| 4 Cannula detection | Tooling + verified pipeline; **blocked on manual labelling** |
| 5 Infection monitoring | Trained |
| 6 Backend + dashboard | Verified end to end |
| 7 Evaluation | Done |
| 8 Deployment | ONNX export, audit logging, checklist |

### Volume

- **6,033 lines** of source (Python) + **601 lines** TypeScript/React
- **1,895 lines** of tests
- **210 tests passing** (180 module + 29 API + 1 slow pipeline test)
- 10 commits, 8 generated documentation reports

### Datasets actually obtained and processed

| Dataset | Content | Use |
|---|---|---|
| CUBITAL | 2,016 NIR images, 1,008 subjects, 2 arms each | Vein training |
| AZH | 830 RGB wound images, 6 classes | Infection training |
| Medetec | 21 IV-relevant images (20 extravasation) | External test |

Split verified **subject-level**: 1,412 / 302 / 302 images from
**706 / 151 / 151 subjects**, zero overlap.

### Measured results — memorise these

| Metric | Value |
|---|---|
| Vein Dice / IoU (test) | **0.3698 / 0.2390** |
| Vein precision / recall | 0.2621 / 0.8141 |
| Infection F1 (5-fold, grouped) | **0.9885 ± 0.0119** |
| Infection AUC | 0.9951 |
| External detection (extravasation, n=21) | **100%**, mean risk 0.993 |
| Latency vein-scan | **654 ms** mean, 667 ms p95 |
| Latency site-check | 6.9 ms mean, 57 ms p95 |
| ONNX parity | 6.6e-06 (tolerance 1e-04) |

### The verified end-to-end demo

Real models, real data, through the API:

```
1. vein-scan     → site (389, 281), angle 23.9°, confidence 0.022
2. normal skin   → risk 0.001, no alert
3. normal skin   → risk 0.001, no alert
4. extravasation → risk 0.998, ALERT "exceeds 0.6; rose 1.00 since last check-in"
5. extravasation → risk 0.999, ALERT
   history       → 4 points, trend "worsening", alert points flagged
```

---

## Rubric 4 · Technical Execution & Code Quality (4 marks)

### This is your strongest section. Lead with bugs you found.

Anyone can show working code. Showing that you **found your own defects by
measurement** is what separates a good project from a demo.

#### Bug 1 — The 0.99 that means almost nothing

The infection model scores F1 0.9885. Rather than reporting it, we ran a feature
ablation:

| Feature group | n | F1 |
|---|---:|---:|
| heterogeneity only | 4 | **0.9876** |
| texture only | 18 | 0.9673 |
| erythema only | 8 | 0.9515 |
| all | 30 | 0.9885 |

**Four colour-variance features alone match all thirty**, and every group clears
0.95. The model separates classes by *visual messiness* — an open wound has red
granulation, yellow slough and black eschar at once; intact skin is uniform.

Early phlebitis is the **opposite** profile: uniformly slightly red *intact*
skin. High erythema, low heterogeneity. So a model leaning on heterogeneity
would miss exactly the case the system exists to catch, while scoring
beautifully on our proxy task.

> Say this out loud: *"Our headline metric is 0.99 and we do not believe it
> means we can detect infection. Here is the ablation that shows why."*

#### Bug 2 — Data leak inflating cross-validation

AZH ships several ROI crops of one wound as `101_0.jpg`, `101_1.jpg`. The venous
class has **173 files but only 118 distinct wounds**. Our original split was
file-level, so crops of the same wound sat in train and test together. Fixed
with `StratifiedGroupKFold` grouped by wound, plus an assertion inside the CV
loop that fails if any group leaks.

*(Honest detail: fixing it barely moved the score — 0.9901 → 0.9885. That is
what led to the ablation, which found the real problem.)*

#### Bug 3 — The model collapsed and the metric hid it

First training run: validation Dice and IoU were **exactly equal** at 0.0559 for
epochs 1, 3 and 4. That identity only holds when true positives are zero.

Val has 17 empty masks out of 302 = 5.63%. An empty mask where the model
predicts nothing scores Dice 1.0. So 17/302 = 0.0563 — the score came *entirely*
from empty masks. A model that had learned **nothing** was reporting 0.056,
which reads like weak learning rather than no learning.

Two fixes: measured the imbalance (278:1), added a `pos_weight` of √278 ≈ 16.7
and shifted the loss to Dice-dominant; **and** changed the metric to report
vein-containing images separately with a `collapsed_to_background` diagnostic.

Result — the second run climbed every epoch:

| Epoch | Run 1 (0.5/0.5, no pos_weight) | Run 2 (0.3/0.7, pos_weight 16.7) |
|---:|---|---|
| 1 | 0.0559 *(collapsed)* | 0.1327 |
| 2 | 0.0495 | 0.1786 |
| 3 | 0.0559 *(collapsed)* | 0.2041 |
| 6 | — | **0.3531** |

#### Bug 4 — A library function that does nothing

`scikit-image` 0.25's `frangi()` returns **byte-identical output** whether
`black_ridges` is True or False. Verified directly: a dark bar and a bright bar
both score 0.579. Without polarity selection the vesselness channel fires on
tendon edges, specular highlights from gel, and stray hairs. We implemented
polarity explicitly against a local-background estimate.

#### Bug 5 — Measured, not assumed

Where should the Frangi filter run? We measured ridge-to-background separation
on a low-contrast vein under sensor noise:

| Input | Separation |
|---|---:|
| raw frame | 11 : 1 |
| CLAHE + bilateral | 40 : 1 |
| **bilateral only** | **201 : 1** |

CLAHE raises contrast in flat skin too, amplifying the speckle Frangi mistakes
for vessels. So the intensity channels keep CLAHE and the vesselness channel
does not.

#### Others worth a one-line mention

- `np.ndarray.ptp()` removed in numpy 2.0
- Vertical veins rejected by a y-on-x line fit (now fits along the longer axis)
- **SIGSEGV**: PyTorch and LightGBM each bundle an OpenMP runtime; loading both
  in one process crashes on macOS
- ROI measured in the uploaded image's frame but used in the *registered*
  image's frame — found via the audit log, before opening a debugger

### Engineering practices to point at

- **210 tests**, including edge cases: empty masks, black ROIs, zero
  registration matches, near-tied candidates
- **Subject-level and patient-level splitting**, asserted in code
- **Audit logging** of every ML call: patient, endpoint, model version, image
  SHA-256, output, latency
- **Model versioning** stored beside every prediction
- **ONNX parity verification** treated as blocking
- **Graceful degradation** — missing model → 503 with instructions, not a 500
- CI running all three suites

---

## Rubric 5 · Presentation (3 marks)

### Order to present in

1. **Problem** (1 min) — two-phase clinical workflow, why each half is hard
2. **Architecture** (2 min) — the diagram above, module boundaries
3. **Live demo** (3 min) — the end-to-end round trip
4. **Results** (2 min) — the metrics table
5. **The 0.99 investigation** (2 min) — *your best material*
6. **Limitations and next steps** (2 min)

### Demo script

```bash
# Terminal 1 — API
cd backend && PYTHONPATH=. ../infection_module/.venv/bin/uvicorn app.main:app --port 8000

# Terminal 2 — dashboard
cd frontend && npm run dev
```

Then open `http://localhost:3000/patients/1`, upload a CUBITAL test image, show
the overlay, upload a normal-skin image then an extravasation image, and show
the alert and the trend chart moving.

**Backup if the demo fails:** `docs/figures/` has saved overlays, and
`docs/full_system_eval.md` has every number. Say "let me show you the saved
output" and keep moving — do not debug in front of the panel.

### One thing to show that nobody else will

Open `docs/figures/vein_example_00.png`. The model has marked an insertion point
**on the bedsheet**, off the patient.

> "This is our current best model, and it puts a needle marker on a bedsheet.
> That's clinically unacceptable and it is not a thresholding problem — creased
> linen has the same ridge geometry as a vein. We traced it to a decision in our
> own data pipeline: CUBITAL's masks label background, arm and vein, and we kept
> only the vein class. Predicting the arm and intersecting constrains every
> candidate to the limb. That's our next change."

A visible failure, a measured cause, and a fix already available in the data.
Volunteering that is worth more than hiding it.

---

## Anticipated questions

**"Is this 30% complete?"**
Roughly 90% of the plan. Seven of eight phases are done; the eighth is blocked
on manual data collection, not on engineering.

**"Your Dice is only 0.37. Isn't that poor?"**
Yes, and there are two reasons. It's epoch 6 of a planned 25 — validation Dice
was still climbing steeply when the machine ran out of swap, so it's a lower
bound. And precision is 0.26 against recall 0.81, meaning it over-predicts onto
background. Both have identified fixes.

**"Why is the infection model 0.99 but the vein model 0.37?"**
Because they are not comparable tasks. The infection number is on a *proxy*
task — wound tissue versus intact skin — which is nearly trivial. The ablation
shows four variance features alone reach 0.988. The vein task is genuinely hard:
0.45% positive pixels, low contrast, real segmentation.

**"How do you know your test set isn't leaking?"**
Three defences. CUBITAL splits by `person_id` from `dataset.csv`, so both arms
of one person stay together — verified 706/151/151 subjects, zero overlap. AZH
groups by wound id, with an assertion inside the CV loop. And the label
validator checks patient leakage for the cannula dataset.

**"What if the camera is RGB, not NIR?"**
Every public vein dataset with masks is NIR — no RGB one exists. Veins are
visible at ~850 nm because haemoglobin absorbs there; in visible light you
mostly see skin surface, and it fails worst on darker skin, which is also where
cannulation is hardest. So NIR for veins, RGB for infection — that's not a
compromise, it's what each task physically requires.

**"Is your training data representative?"**
No, and it's our biggest limitation. CUBITAL is ages **5–22, median 11**, and
**97% recorded as slim** — 42 obese subjects out of 2,016. The clinical need is
elderly and obese adults. Gender is balanced (1040 F / 976 M), which is the one
axis that is representative.

**"Why didn't you finish the cannula detector?"**
No open-access dataset of IV cannula sites exists — we checked IEEE DataPort
(IV infusion *containers*, not sites) and Roboflow (ambulance equipment and
laparoscopic trocars, different objects). It needs 100–200 photographs we have
to take ourselves. We deliberately did not fabricate images: a detector trained
on invented data would score well and fail on a real arm. Instead we built a
validator and verified the training pipeline, so labelling is the only work left.

**"Can this be used on patients?"**
No. It is a research prototype, not a medical device, not validated. Real use
requires IRB approval and a formal validation study. The system is advisory by
design — there is no endpoint that performs an action on a patient.

**"What's the single most valuable next step?"**
Using CUBITAL's arm class, which we currently discard. It makes off-patient
predictions impossible by construction and should raise precision sharply. One
day of work, no new data.

---

## Numbers to have on the tip of your tongue

| | |
|---|---|
| Model parameters | 7,850,573 (<15M budget) |
| Vein pixels per frame | 0.358% → **278:1** imbalance |
| pos_weight used | √278 ≈ **16.7** |
| Training split | 1,412 / 302 / 302 images, 706 / 151 / 151 subjects |
| Vein Dice / IoU | 0.3698 / 0.2390 |
| Infection F1 | 0.9885 |
| Ablation punchline | **4 features ≈ 30 features** |
| AZH leak | 173 files, 118 wounds |
| Latency | 654 ms / 6.9 ms |
| Tests | 210 |

---

## Traps to avoid

- **Do not quote 0.99 as an achievement.** Present it with the ablation. If you
  say "our model detects infection with 99% accuracy" and they probe, you lose
  more than the mark.
- **Do not say "certified nurses".** The CUBITAL certificates are signed
  *Licenciada en Bioquímica* — biochemists, who do perform phlebotomy. The
  paper's abstract is wrong. Say "qualified phlebotomy practitioners".
- **Do not claim the vein model is finished.** Epoch 6 of 25.
- **Do not hide the bedsheet failure.** It is your best slide.
- **Do not debug live.** Fall back to saved figures.

---

## If you are asked "what did you learn"

> That the hardest part wasn't building the models — it was working out which of
> our numbers to trust. Our best-looking result, F1 0.99, turned out to be
> measuring visual messiness rather than redness, and we only found that by
> ablating our own features. Our worst-looking result, Dice 0.37, is a genuinely
> hard task and an honest lower bound. Learning to tell those apart mattered
> more than any single model.
