# Dataset Sources

Availability verified 2026-08-19. Two of the four datasets are directly
downloadable; one requires an author request; one requires scraping.

| Dataset | Modality | Access | Masks? | Module |
|---|---|---|---|---|
| CUBITAL | NIR | Public (Google Drive) | Yes | Pre-cannulation vein |
| U-DAVIS | NIR | Request from authors | Yes | Pre-cannulation vein |
| AZH Wound | RGB | Public (GitHub) | Type labels, no infection labels | Post-cannulation infection |
| Medetec | RGB | Public (scrape) | Category labels only | Post-cannulation infection |

---

## CUBITAL — NIR forearm vein dataset

**Source:** <https://github.com/EdwinTSalcedo/CUBITAL>
**Download:** <https://drive.google.com/drive/folders/19DaK7T81qTxgBzirvBGdMyVUj8mOCcOx>
**Paper:** Salcedo et al., *Edge AI-Based Vein Detector for Efficient Venipuncture
in the Antecubital Fossa* — <https://arxiv.org/abs/2310.18234>

2,016 near-infrared images of the antecubital fossa from 1,008 individuals (two
samples per subject, one per arm). Ships in four versions: **A** (base, full
annotations), **B** (augmented), **C** (resized to 512×512), **D** (resized +
augmented). Masks are single-channel greyscale with pixel values `0` =
background, `1` = arm, `2` = vein — so vein masks must be extracted as
`mask == 2`, *not* thresholded at 127. A demographics CSV accompanies the
images with age, complexion, gender, arm angle, and antecubital fossa
coordinates.

**Why it matters most:** a separate validation folder holds **384 samples
annotated by three certified nurses** marking where they would actually perform
venipuncture. This is the ground truth for the Phase 3 insertion-point distance
metric — no other dataset here provides it.

**Use in this project:** primary training set for the Attention U-Net vein
segmenter (Phase 3). Version C matches our 512×512 target, so it is the default.
Because there are two images per subject, splitting must be **subject-level**
(Phase 2) or the same arm leaks across train and test.

---

## U-DAVIS — arm venous NIR dataset

**Paper:** Kuthiala et al. (2022), *U-DAVIS: Deep Learning Based Arm Venous Image
Segmentation Technique for Venipuncture*, Computational Intelligence and
Neuroscience — <https://doi.org/10.1155/2022/4559219>
(PubMed: <https://pubmed.ncbi.nlm.nih.gov/36238666/>)

1,850 NIR forearm images at 2592×1944 (5 MP), JPG, collected from 450
individuals aged 25–75 at the Northern Railway Health Unit, Ludhiana, India,
under controlled ambient lighting inside an enclosure. Annotated with
corresponding vein masks.

**Access caveat:** no public download mirror was found. The dataset must be
requested from the corresponding author at Thapar Institute of Engineering and
Technology. **Budget 1–3 weeks for a reply, and treat it as optional** — Phase 3
must be able to train on CUBITAL alone.

**Use in this project:** secondary training data and a cross-source
generalization check (train on CUBITAL, test on U-DAVIS) once access is granted.
Its much higher native resolution also makes it the better source if we later
want to evaluate at above 512×512.

---

## AZH Wound and Vascular Center dataset

**Source:** <https://github.com/uwm-bigdata/wound-classification-using-images-and-locations>
**Related:** <https://github.com/uwm-bigdata/wound-segmentation>
**Paper:** *Multi-modal wound classification using wound image and location by
deep neural network* — <https://pmc.ncbi.nlm.nih.gov/articles/PMC9681740/>

730 pre-filtered ROI images plus 538 whole-wound RGB images, collected over two
years at the AZH Wound and Vascular Center (Milwaukee, WI) with an iPad Pro and
a Canon SX 620 HS, labelled by a wound specialist. Variable size: width
320–700 px, height 240–525 px. JPG.

**Label caveat that shapes Phase 5:** labels are **wound type** — venous,
diabetic, pressure, surgical — **not infected/uninfected**. There is no
ground-truth infection label in this dataset. Any `is_infected` column derived
from it is a *proxy* label and must be documented as such in the Phase 5 model
report, since it caps how much the reported precision/recall can be trusted.

**Use in this project:** trains the colour + texture feature extractor and the
LightGBM risk model (Phase 5). Its value is that it contains genuine
inflamed/erythematous tissue photographed on real patients across skin tones,
which is the visual signal the risk model needs to learn.

---

## Medetec Wound Database

**Source:** <https://www.medetec.co.uk/files/medetec-image-databases.html>
**Stock images index:** <https://www.medetec.co.uk/files/medetec-images.html>

Roughly 594 RGB images across ~15 wound categories, published by Medetec as
free stock images. Delivered as browsable per-category HTML gallery pages rather
than a single archive, so acquisition means scraping the category index pages.
Licensing is described on the site as free stock/copyright-free but is not
stated as a formal SPDX licence — **confirm terms before any redistribution**;
using it locally for model training is fine, re-hosting the images is not.

**Category worth calling out:** Medetec includes an
[extravasation injuries](https://www.medetec.co.uk/slide%20scans/extravasation-wound-images/index.html)
gallery. Extravasation is an IV-infiltration injury — this is the closest
publicly available imagery to the actual failure mode IntelliCannula monitors
for, and is more on-target than any other data listed here.

**Use in this project:** held out as an **external generalization test set**
(`data/processed/infection/medetec_external_test/`), never mixed into the AZH
training split. Different cameras, different era, different clinicians — so
performance here is the honest estimate of how the risk model travels.

---

## Gaps this table does not close

1. **No public RGB vein dataset with masks exists.** Both vein datasets are NIR.
   This is the single fact driving the camera-modality decision — see
   `README.md` § Camera modality.
2. **No public cannula-detection dataset.** Phase 4 needs ~100–200 hand-labelled
   images; see `docs/labeling_guide.md` (written in Phase 4).
3. **No public longitudinal IV-site dataset.** Nothing here has the same site
   photographed over consecutive days, which is what the Phase 5 temporal
   tracker actually models.
