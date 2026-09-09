# Cannula Hub Labelling Guide

Phase 4 needs a dataset that does not exist publicly. Budget **1–2 days** to
produce it before starting that phase.

## What to draw

A tight box around the **visible cannula hub / dressing junction** — the plastic
connector where the cannula enters the dressing. **Not** the whole dressing, and
**not** the whole arm.

Why tight: the box anchors the infection-monitoring ROI, and `crop_roi` expands
it by a proportional margin (default 40% per side). A box drawn around the whole
dressing would expand to include background and clothing, and the healthy-skin
reference patch would be sampled from a bedsheet.

Single class, id `0`, name `cannula_hub`.

## How many, and of what

**100–200 images.** Below ~100, even the nano model overfits. What matters more
than raw count is spread across:

| Vary | Why |
|---|---|
| **Skin tone** — aim for even coverage across the full range | A detector trained on light skin fails on dark skin. This is the single most common fairness failure in medical imaging, and it is entirely avoidable at labelling time. |
| **Arm angle and rotation** | Photos will be taken bedside, not in a rig. |
| **Lighting** — daylight, fluorescent ward light, phone flash | Colour temperature shifts dramatically between these. |
| **Dressing type and condition** | Transparent film, gauze, bloodstained, lifting at an edge. |
| **Distance** — close-up to arm's length | Trains scale invariance. |

Deliberately include **hard negatives**: sites with no cannula, other tubing,
watches and bracelets, tape without a hub. Without them the detector learns
"find the shiny plastic thing" and fires on a wristwatch.

## Tools

[Roboflow](https://roboflow.com) (easiest, free tier, exports YOLO directly) or
[CVAT](https://www.cvat.ai) (self-hosted, no upload of patient images to a third
party — **use this if the images are not fully de-identified**).

## Format

One `.txt` per image, same stem, in `labels/` alongside `images/`:

```
0 x_center y_center width height
```

All four values **normalised to [0, 1]** relative to image dimensions. One line
per instance; an image with no cannula gets an empty `.txt` file (not a missing
one — a missing file is ambiguous between "no object" and "not yet labelled").

## Splitting

Split **70/15/15 by patient, not by image**, before export. If you photograph
one site three times, all three must land in the same split — otherwise the test
mAP measures memorisation.

This is the same trap that inflated the Phase 5 risk model to F1 0.99: AZH ships
several crops of one wound, and an ungrouped split put them in both train and
test. See `infection_module/src/data_loader.py::wound_group_id`.

## Before you photograph anything

Patient photographs are identifiable data. Confirm consent and your
institution's imaging policy first, strip EXIF (it carries GPS and device
serials), and do not upload to a cloud labelling service unless your institution
has approved that specific service.

## Where files go

```
infection_module/data/cannula_raw/
  images/   <- source photographs
  labels/   <- YOLO .txt files, matching stems
```

After export, arrange into `data/cannula/{train,val,test}/{images,labels}/` to
match `infection_module/src/cannula_detector/data.yaml`.

## Validate before you train

```bash
python infection_module/src/cannula_detector/validate_labels.py --root data/cannula
```

Run this **first**. Every check it makes corresponds to a failure that would
otherwise waste the whole training run, or silently produce an inflated score:

| Check | Why it matters |
|---|---|
| Pixel vs normalised coordinates | The most common export mistake. YOLO wants `[0,1]`, and pixel values train a model on nonsense. |
| Missing label files | A missing `.txt` is ambiguous between "no cannula" and "not yet labelled". Create an empty file for genuine negatives. |
| Class id ≠ 0 | Single-class dataset; a stray id 1 silently creates a phantom class. |
| Whole-frame boxes | `crop_roi` expands the box 40% per side. A box round the entire dressing pulls in bedsheet, and the healthy-skin reference gets sampled from linen. |
| **Patient leakage across splits** | The big one. Three photos of one site in train *and* test means test mAP measures memorisation. This exact trap inflated the Phase 5 risk model to F1 0.99 before it was caught. |
| No negative images | Without hard negatives the detector learns "find the shiny plastic thing" and fires on a wristwatch. |

Then:

```bash
python infection_module/src/cannula_detector/train.py
```

The training pipeline itself is already verified — `test_cannula_pipeline.py`
runs a two-epoch train/checkpoint/infer cycle against a public sample dataset,
so when your images are ready, training fails for dataset reasons only, never
because of a typo in the training call.
