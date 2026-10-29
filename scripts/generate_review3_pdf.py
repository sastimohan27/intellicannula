"""Generate the Review 3 implementation document, following Project_Doc_Template.pdf section by section.

    <reportlab venv>/bin/python scripts/generate_review3_pdf.py [output.pdf]

Figures come from docs/figures/review3/ (built by scripts/build_review3_figures.py
from the live system). Every number below is taken from a measured artifact in
docs/ or from a run recorded in Appendix C; nothing is estimated.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus import Image as RLImage

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/figures/review3"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/PROJECT_REVIEW_3_IMPLEMENTATION_DOCUMENTATION.pdf"

SUP = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Body", str(SUP / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-B", str(SUP / "Arial Bold.ttf")))
pdfmetrics.registerFont(TTFont("Body-I", str(SUP / "Arial Italic.ttf")))
pdfmetrics.registerFont(TTFont("Body-BI", str(SUP / "Arial Bold Italic.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")
pdfmetrics.registerFont(TTFont("Mono", str(SUP / "Courier New.ttf")))
# DejaVu Sans provides the checkbox glyph (U+2611); it ships with matplotlib.
_dejavu = next(p for p in [ROOT / "infection_module/.venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"] if p.exists())
pdfmetrics.registerFont(TTFont("Sym", str(_dejavu)))

NAVY, BLUE, GRID, BOXBG = "#17335B", "#1F4E79", "#222222", "#F2F2F2"
LM = 62
W = 612 - 2 * LM  # 488


def S(name, **kw):
    base = dict(fontName="Body", fontSize=8.6, leading=11.6, textColor=colors.HexColor("#333333"))
    base.update(kw)
    return ParagraphStyle(name, **base)


body = S("body")
bodyi = S("bodyi", fontName="Body-I")
cell = S("cell", fontSize=7.8, leading=10, textColor=colors.HexColor("#444444"))
cellb = S("cellb", fontSize=7.8, leading=10, fontName="Body-B", textColor=colors.HexColor("#222222"))
cellc = S("cellc", fontSize=7.8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
hdr = S("hdr", fontName="Body-B", fontSize=8, leading=10, textColor=colors.white, alignment=TA_CENTER)
mono = S("mono", fontName="Mono", fontSize=7.4, leading=9.4, textColor=colors.HexColor("#333333"))
h1 = S("h1", fontName="Body-B", fontSize=15, leading=19, textColor=colors.HexColor("#1B365D"), spaceBefore=12, spaceAfter=6, keepWithNext=1)
h2 = S("h2", fontName="Body-B", fontSize=11, leading=14, textColor=colors.HexColor(BLUE), spaceBefore=8, spaceAfter=4, keepWithNext=1)
note = S("note", fontName="Body-I", fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"), spaceAfter=4)
cap = S("cap", fontName="Body-I", fontSize=8.2, leading=10.4, alignment=TA_CENTER, textColor=colors.HexColor("#333333"), spaceBefore=2, spaceAfter=7)

story: list = []
MODEL: list = []   # neutral copy of the content, rendered to Word by generate_review3_docx.py


def P(t, st=body):
    return Paragraph(t, st)


def NOTE(t):
    MODEL.append(("note", t))
    story.append(P(t, note))


def H1(t):
    MODEL.append(("h1", t))
    story.append(P(t, h1))


def H2(t):
    MODEL.append(("h2", t))
    story.append(P(t, h2))


def shaded(paras, st=body):
    """Grey content panel, one row per paragraph so it can split across pages."""
    MODEL.append(("shaded", list(paras), st is bodyi))
    rows = [[p if not isinstance(p, str) else P(p, st)] for p in paras]
    t = Table(rows, colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BOXBG)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    story.append(t)
    story.append(Spacer(1, 5))


def table(header, rows, widths, centre_cols=(), mono_cols=(), bold_first=False, fs=None):
    MODEL.append(("table", list(header), [list(r) for r in rows], list(widths), list(centre_cols), list(mono_cols), bold_first))
    data = [[P(h, hdr) for h in header]]
    for r in rows:
        line = []
        for i, c in enumerate(r):
            if isinstance(c, str):
                st = cellc if i in centre_cols else (cellb if (bold_first and i == 0) else cell)
                if i in mono_cols:
                    st = S("m", fontName="Mono", fontSize=7.2, leading=9.2, textColor=colors.HexColor("#333333"))
                c = P(c, st)
            line.append(c)
        data.append(line)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(t)
    story.append(Spacer(1, 6))


def figure(file, caption, max_w=W, max_h=330):
    path = FIG / file
    MODEL.append(("figure", str(path), caption, max_w, max_h))
    w, h = Image.open(path).size
    sc = min(max_w / w, max_h / h)
    story.append(KeepTogether([RLImage(str(path), width=w * sc, height=h * sc), P(caption, cap)]))


def steps(prefix, items):
    MODEL.append(("steps", list(items)))
    rows = [[P(f"<i>Step {i}: {s}</i>", S("st", fontName="Body-I", fontSize=8, leading=10.5, textColor=colors.HexColor("#444444")))] for i, s in enumerate(items, 1)]
    t = Table(rows, colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BOXBG)), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    story.append(t)
    story.append(Spacer(1, 5))


# ============================================================ page chrome
class Canvas(canvas.Canvas):
    def showPage(self):
        self.saveState()
        self.setFont("Body-B", 7.6)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawRightString(612 - LM + 8, 792 - 40, "PROJECT REVIEW 3 | IMPLEMENTATION DOCUMENTATION")
        self.setFont("Body", 7.6)
        self.drawCentredString(306, 30, "Review 3 - 100% Implementation Required")
        self.restoreState()
        super().showPage()


doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=LM, rightMargin=LM, topMargin=62, bottomMargin=54,
                      title="Project Review 3 - Implementation Documentation - IntelliCannula",
                      author="Sastika M, Akshaya GK")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, 54, W, 792 - 62 - 54, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)])])

# ============================================================ title page
story += [Spacer(1, 70),
          P("PROJECT REVIEW 3", S("t", fontName="Body-B", fontSize=30, leading=36, alignment=TA_CENTER, textColor=colors.HexColor("#1B365D"))),
          Spacer(1, 8),
          P("IMPLEMENTATION DOCUMENTATION TEMPLATE", S("t2", fontName="Body-B", fontSize=17, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#1F4E79"))),
          Spacer(1, 22)]
band = Table([[P("REVIEW 3 REQUIREMENT: 100% IMPLEMENTATION COMPLETED", S("b", fontName="Body-B", fontSize=11.5, alignment=TA_CENTER, textColor=colors.HexColor("#1B365D")))]], colWidths=[W - 40])
band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#D9EAF5")), ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 11)]))
story += [band, Spacer(1, 20)]
meta = [("Project Title", "IntelliCannula: Two-Module Clinical Decision Support for Peripheral IV Cannulation (Pre-Cannulation Vein Guidance &amp; Post-Cannulation Infection Monitoring)"),
        ("Student Name(s)", "Sastika M, Akshaya GK"),
        ("Registration Number(s)", "23MIA1047, 23MIA1127"),
        ("Programme / Branch", "M.Tech. Integrated Computer Science and Engineering (Business Analytics)"),
        ("Course / Project Code", "SWE1017 — Project Review 3 (Final Implementation Review)"),
        ("Faculty Guide", "Dr. Manjula V. (Professor, School of Computer Science and Engineering)"),
        ("Review", "Review 3 - Final Implementation Review")]
mt = Table([[P(f"<b>{a}</b>", S("ml", fontName="Body-B", fontSize=9.5, leading=13, textColor=colors.HexColor("#1B365D"))), P(b, S("mv", fontSize=9.5, leading=13, textColor=colors.HexColor("#333333")))] for a, b in meta], colWidths=[150, W - 190])
mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9DB7CC")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9DB7CC")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
story += [mt, PageBreak()]

# ============================================================ checklist
story.append(P("Review 3 Submission Checklist", S("ch", fontName="Body-B", fontSize=15, leading=19, textColor=colors.HexColor("#1B365D"), spaceAfter=8)))
items = ["Complete source code submitted", "All modules integrated and executable", "Dataset / database / sample input included or access provided",
         "Dependencies / requirements file included", "README / execution instructions included", "Screenshots and outputs included",
         "Testing and validation completed", "Results / performance metrics reported", "Documentation completed"]
box_style = S("bx", fontName="Sym", fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
data = [["", P("Item", hdr), P("Status", hdr)]] + [[P("☑", box_style), P(i, S("ci", fontSize=8.6, textColor=colors.HexColor("#555555"))), P("Yes", S("cy", fontName="Body-I", fontSize=8.6, textColor=colors.HexColor("#333333")))] for i in items]
ct = Table(data, colWidths=[28, 340, 120])
ct.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
story += [ct, Spacer(1, 8),
          P("Submission notes: the trained checkpoints (<font name='Mono'>vein_module/models/</font>, <font name='Mono'>infection_module/models/</font>), the raw/processed datasets and the demo images (<font name='Mono'>data/demo/</font>) are git-ignored because of their size, so they are supplied in the submission folder next to the repository. The public datasets can be re-downloaded with <font name='Mono'>scripts/download_datasets.py</font> (Section 13).", note),
          PageBreak()]

# ============================================================ contents
story.append(P("Contents", S("ct", fontName="Body-B", fontSize=15, leading=19, textColor=colors.HexColor("#1B365D"), spaceAfter=8)))
for line in ["1. Problem Statement and Objectives", "2. Finalized System Architecture / Methodology", "3. Module-wise Implementation",
             "4. Algorithms, Models and Technologies Used", "5. Dataset / Database Details", "6. Implementation Screenshots", "7. Experimental Setup",
             "8. Testing and Test Cases", "9. Results and Performance Metrics", "10. Comparison and Analysis", "11. Limitations", "12. Conclusion",
             "13. Instructions for Executing the Project", "Appendix A. Source Code Structure", "Appendix B. Dependencies and Configuration", "Appendix C. Additional Evidence"]:
    story.append(P(line, S("cl", fontSize=10, leading=17, leftIndent=24, textColor=colors.HexColor("#1B365D"))))
story.append(PageBreak())

# ============================================================ 1
H1("1. Problem Statement and Objectives")
H2("1.1 Background / Context")
shaded(["Peripheral intravenous cannulation (PIVC) is among the most common invasive procedures in hospital care; more than a billion peripheral catheters are reported to be placed each year. Two problems recur. <b>Before insertion</b>, clinicians must find a usable vein by sight and touch; in patients with difficult venous access (children, the elderly, obese or dehydrated patients, darker skin) first-attempt failure is commonly reported at 30–50 %, and repeated attempts cause pain, vessel damage and treatment delay. <b>After insertion</b>, a sizeable share of catheters fail early through phlebitis, infiltration or infection, and the bedside check is a subjective visual inspection in which early, mild redness is easily missed or cannot be compared with how the site looked the day before."])
H2("1.2 Problem Statement")
shaded(["There is no integrated, auditable decision-support tool that (a) helps choose an insertion site before cannulation and (b) measures change at the cannula site against the patient's own baseline afterwards. Commercial vein illuminators only project an image; they do not segment veins, grade them, or suggest a site. Naive computer-vision pipelines also fail silently in this setting, and each of the following failures was observed and fixed during this project:",
        "(1) A vein network trained on near-infrared (NIR) images produced confident vein masks on ordinary RGB phone photos, which carry no NIR vein contrast. (2) A wound-versus-skin classifier used as an infection score rated every site photo ≈ 1.00, and rated a landscape photo 0.999, because it measured tissue heterogeneity and not redness. (3) Day-to-day photos differ in lighting, white balance and camera position, which confounds any comparison across days.",
        "IntelliCannula addresses these with refusal gates and a baseline-anchored measurement, under one rule: <i>the system recommends, it never acts</i>. It is a research prototype, not a medical device."])
H2("1.3 Project Objectives")
NOTE("List measurable and implementation-oriented objectives.")
shaded(["<b>Objective 1 (NIR provenance gate):</b> the vein model must never run on an RGB or unverified image: accept only frames whose SHA-256 matches the CUBITAL NIR dataset or that come from a registered NIR device with a matching token; reject phone photos (camera EXIF) and everything else with HTTP 422.",
        "<b>Objective 2 (vein segmentation):</b> train an Attention U-Net (7.85 M parameters) with a weighted BCE + soft-Dice loss on subject-disjoint CUBITAL splits and reach test Dice ≥ 0.55 (achieved 0.567).",
        "<b>Objective 3 (insertion-site selection):</b> from the vein mask, deterministically select a straight run (sliding-window R² ≥ 0.95), measure vein width, and return an insertion point, a suggested needle angle bounded to 15°–30° and a grade; refuse when no segment qualifies.",
        "<b>Objective 4 (site monitoring):</b> verify the cannula hub with YOLOv11 (confidence ≥ 0.50), verify the follow-up shows the same scene as the baseline (≥ 50 RANSAC inliers), measure redness (CIELAB Δa*) in a skin ring against the patient's baseline, and raise advisory alerts; the change measurement must track known injected redness with r ≥ 0.99 in a controlled validation (achieved 0.995)."])
H2("1.4 Scope of the Project")
shaded(["<b>In scope:</b> NIR forearm vein segmentation and site selection; NIR provenance gate; photo-quality and cannula gates; multi-day scene registration; baseline-anchored redness measurement and alert rules; FastAPI back end with PostgreSQL/SQLite persistence and audit logging; Next.js clinician dashboard; automated tests and measured evaluation.",
        "<b>Out of scope:</b> autonomous needle insertion; diagnosis of infection or phlebitis (redness is advisory and cannot confirm or exclude infection); clinical validation on patients; vein scanning with consumer RGB cameras (physically unsupported); user authentication (the prototype accepts synthetic or de-identified data only)."])

# ============================================================ 2
H1("2. Finalized System Architecture / Methodology")
H2("2.1 Overall System Architecture")
shaded(["The system has four layers (Figure 1). The <b>clinician dashboard</b> (Next.js) talks to the <b>FastAPI server</b> over REST and a WebSocket alert channel. The server performs upload validation, the NIR modality gate, the photo-quality gate and audit logging, and calls two in-process ML modules whose models are loaded once and cached: the <b>vein module</b> (M1: NIR provenance → preprocessing → Attention U-Net → site selector) and the <b>site module</b> (M2/M3: quality gate → YOLOv11n cannula gate → ORB-RANSAC scene check → CIELAB redness and alert rules). Results are persisted through SQLAlchemy to PostgreSQL (automatic SQLite fallback) with images in a local or S3/MinIO store."])
figure("fig1_architecture.png", "Figure 1. Finalized system architecture", max_h=300)
H2("2.2 End-to-End Workflow / Methodology")
shaded(["<b>Before insertion</b> a clinician uploads an NIR forearm frame. The provenance gate checks it; a verified frame is preprocessed (CLAHE, bilateral filter, Frangi vesselness), segmented, post-processed (threshold 0.90, components under 60 px removed) and passed to the site selector, which returns the insertion point, vein width, suggested angle and grade, plus an overlay. <b>After insertion</b>, a daily RGB photo of the site passes the quality gate and the cannula gate. The first accepted photo becomes the patient's baseline; each follow-up is aligned to it (ORB + RANSAC, ≥ 50 inliers), colour-matched on white/grey objects away from the site, and scored for redness in the skin ring around the hub. The score, the change since baseline and the alert rules produce a status; the result is stored, written to the audit log and pushed to the dashboard. Every gate refuses rather than guesses (red boxes in Figure 2)."])
figure("fig2_workflow.png", "Figure 2. End-to-end project workflow / methodology", max_h=270)

# ============================================================ 3
H1("3. Module-wise Implementation")
NOTE("Summarize all implemented modules. Every proposed module should appear in this table.")
table(["Module No.", "Module / Feature", "Purpose", "Implementation Status", "Evidence"], [
    ["M1", "Pre-cannulation vein guidance", "NIR provenance gate, Attention U-Net segmentation, insertion-site, angle and grade", "100% Complete", "vein_module/, backend/app/api/vein.py, Fig 3, 8"],
    ["M2", "Cannula localisation and quality gate", "Photo-quality screening and YOLOv11n cannula hub verification (anchors the site ROI)", "100% Complete", "infection_module/src/cannula_detector/, backend/app/core/ml.py, Fig 4"],
    ["M3", "Longitudinal redness and alert engine", "Scene registration, baseline-anchored CIELAB redness measurement, status and alert rules", "100% Complete", "infection_module/src/erythema.py, registration.py, temporal_tracker.py, Fig 5, 11"],
    ["M4", "Clinician dashboard and audit API", "Next.js UI, REST/WebSocket API, patient history, persistence, structured audit log", "100% Complete", "frontend/, backend/app/, Fig 7–10"]],
    [38, 100, 150, 70, 130], centre_cols=(0, 3))

H2("3.1 Module 1: Pre-Cannulation Vein Guidance")
shaded(["<b>Purpose and function.</b> Find usable veins in an NIR forearm image and recommend an insertion site, with the width, suggested angle and grade behind it.",
        "<b>Inputs.</b> One NIR forearm image, plus either a matching SHA-256 in the local CUBITAL registry or the headers <font name='Mono'>X-NIR-Device</font> / <font name='Mono'>X-NIR-Device-Token</font> of a registered device.",
        "<b>Processing.</b> (1) <i>Provenance gate</i> (<font name='Mono'>backend/app/core/modality.py</font>): pixel statistics cannot separate NIR from RGB here (CUBITAL chroma spread overlaps RGB photos), so acceptance is by provenance; camera EXIF → rejected as RGB, anything else unverified → HTTP 422, and the model is never called. (2) <i>Preprocessing</i> (<font name='Mono'>vein_module/src/preprocessing.py</font>): CLAHE (clip 2.0, 8×8) on L of LAB with bilateral smoothing, and a polarity-filtered Frangi vesselness map (σ = 1,3,5,7,9; dark ridges only); the network input is [preprocessed R, preprocessed G, Frangi]. (3) <i>Segmentation</i>: Attention U-Net, 7,850,573 parameters, sigmoid output thresholded at 0.90, connected components under 60 px removed. (4) <i>Site selection</i> (<font name='Mono'>site_selector.py</font>): skeletonise, measure width by distance transform, keep runs that are straight (sliding 9-px window, R² ≥ 0.95), ≥ 3 px wide and ≥ 15 px long; score = 0.5·width + 0.3·length + 0.2·(distance from edge); confidence is the margin over the runner-up (a single candidate scores 0.5); the angle is a heuristic proxy for depth, clamped to 15°–30°.",
        "<b>Outputs.</b> Binary mask, overlay, insertion point (x, y), vein width (px), angle (°), grade, confidence, reason, model version, image SHA-256, inference time. Result of the recorded live run (Figure 3): site (269, 238), width 4.24 px, 25.1°, Grade C (Difficult), confidence 0.50 (single candidate), verified as a CUBITAL test-split frame."])
figure("fig3_module1.png", "Figure 3. Module 1 implementation / output screenshot: verified NIR input, Frangi channel, predicted mask and the selected insertion site (live API output)", max_h=190)

H2("3.2 Module 2: Cannula Localisation and Quality Gate")
shaded(["<b>Purpose and function.</b> Make sure a site photo is usable and that it really shows a cannula, and anchor the measurement region on the hub.",
        "<b>Inputs.</b> One RGB site photo (phone) with the hub in frame.",
        "<b>Processing.</b> <i>Quality gate</i> (<font name='Mono'>backend/app/core/ml.py</font>): short side ≥ 256 px, Laplacian variance ≥ 20 (measured on ≤ 1024 px; sharp photos score ≥ 57, blurred copies ≤ 3.3), mean brightness ≥ 40, overexposed pixels ≤ 25 %; otherwise a retake is requested before any model runs. <i>Cannula gate</i>: YOLOv11n (Ultralytics) fine-tuned for 15 epochs at 640 px on 90 synthetic composites (a procedural hub on real skin backgrounds); the model is loaded once and cached in the process. Confidence ≥ 0.50 → verified; 0.25–0.50 → “unable to verify” (refused); &lt; 0.25 → “no cannula” (refused); nothing is stored and no score is produced. On a <i>follow-up</i> whose baseline hub was verified and whose scene matches, a weak detection (0.25–0.50) is accepted and the hub position is taken from the baseline (red skin lowers detector confidence).",
        "<b>Outputs.</b> Hub bounding box, detector confidence, verification state (<font name='Mono'>detected</font> / baseline position), or an HTTP 422 with a retake instruction. The recorded run (Figure 4) verified the hub at confidence 0.71 and refused a landscape photo (0.24)."])
figure("fig4_module2.png", "Figure 4. Module 2 implementation / output screenshot: cannula gate accepting a hub photo and refusing a photo with no cannula (live API output)", max_h=215)

H2("3.3 Module 3: Longitudinal Redness and Alert Engine")
shaded(["<b>Purpose and function.</b> Measure how much redder the skin around the cannula is than at baseline, robustly to lighting and camera movement, and raise advisory alerts.",
        "<b>Inputs.</b> The current photo, the patient's baseline photo and the verified hub box.",
        "<b>Processing.</b> (1) <i>Scene check</i> (<font name='Mono'>registration.py</font>): 2,000-feature ORB matching, Lowe ratio 0.75, RANSAC homography; fewer than 50 inliers → refused as a different scene (same scene 881–1,865 inliers, unrelated ≤ 11). (2) <i>Skin ring</i> (<font name='Mono'>erythema.py</font>): YCrCb skin mask (grey tape, dressing and the hub are excluded), ring from 0.75 to 5.0 hub half-sizes around the hub. (3) <i>Redness</i>: a* in CIELAB; reference = median a* of the farthest 30 % of visible skin in the same photo; a pixel counts as reddened when Δa* &gt; 6; <i>risk score</i> = reddened fraction / 0.5, capped at 1. (4) <i>Change since baseline</i>: align the follow-up to the baseline, match colours on white/grey objects outside the ring, and take the share of baseline-ring skin whose a* rose by ≥ 6. (5) <i>Status and alerts</i> (<font name='Mono'>temporal_tracker.py</font>): alert when the score &gt; 0.6 (≈ 30 % of ring skin reddened), or rises by &gt; 0.15 since the last check-in, or increases three check-ins in a row. Statuses: Baseline recorded · No significant visible change · Visible change detected (change ≥ 10 pts or level ≥ 2) · Review recommended (alert) · Unable to compare (for example no white/grey reference in frame).",
        "<b>Outputs.</b> Score, reddened fraction, redness level, change since baseline, scene inliers, status, plain-language explanation, alert flag and reason, trend. In the recorded run (Figure 5) the scores were 0.11 → 0.21 → 0.82 and the third check-in raised an alert (score &gt; 0.6 and a +0.61 jump)."])
figure("fig5_module3.png", "Figure 5. Module 3 implementation / output screenshot: baseline and two follow-ups of a simulated sequence with the change map and alert state (live API output)", max_h=240)

H2("3.4 Integration of Modules")
shaded(["<b>API.</b> The dashboard calls <font name='Mono'>POST /patients</font>, <font name='Mono'>POST /patients/{id}/vein-scan</font> (201, or 422 when not verified NIR), <font name='Mono'>POST /patients/{id}/site-check</font> (201, or 422 when the photo or cannula is rejected), <font name='Mono'>GET /patients/{id}/history</font>, <font name='Mono'>GET /modality/nir</font>, <font name='Mono'>GET /models</font>, and listens on the WebSocket <font name='Mono'>/ws/alerts</font>. ML endpoints return 503 with the missing package named if the ML environment is absent, never a fabricated result.",
        "<b>Execution flow.</b> An upload is validated, hashed, passed through the gates, run through the cached model(s), stored through SQLAlchemy (tables <font name='Mono'>patients</font>, <font name='Mono'>vein_scans</font>, <font name='Mono'>site_checkins</font>; images in the local store or S3/MinIO) and recorded as one structured JSON audit line (timestamp, endpoint, patient, model version, image SHA-256, status, latency). Alerts are broadcast on the WebSocket.",
        "<b>M4 – dashboard.</b> Next.js 16 / React 19 / Tailwind 4: patient directory with alert and “site check due” badges, a patient page with the NIR upload, the site-photo capture, the redness-over-time chart (alert line at 0.6) and a history table, and a model panel fed live from <font name='Mono'>/models</font> that states what is and is not validated. A banner states that the system is advisory."])

# ============================================================ 4
H1("4. Algorithms, Models and Technologies Used")
table(["Category", "Algorithm / Model / Technology", "Version / Configuration", "Purpose in Project"], [
    ["Programming Language", "Python; TypeScript", "Python 3.12.8; TypeScript 5, Node v24.14.1", "ML, API; dashboard"],
    ["Framework / Library", "PyTorch; torchvision", "torch 2.14.0, Apple MPS", "Attention U-Net training and inference"],
    ["Framework / Library", "OpenCV; scikit-image; SciPy; NumPy", "5.0.0; 0.25.0; 1.18.1; 2.1.3", "CLAHE, bilateral, Frangi, skeleton, distance transform, ORB/RANSAC, CIELAB"],
    ["Algorithm / Model", "Attention U-Net", "7,850,573 params; weighted BCE 0.3 + Dice 0.7; AdamW 1e-4", "NIR vein segmentation"],
    ["Algorithm / Model", "YOLOv11n (Ultralytics)", "ultralytics 8.4.123; 640 px, 15 epochs; gate ≥ 0.50", "Cannula hub detection"],
    ["Algorithm / Model", "ORB + RANSAC; CIELAB Δa*", "ratio 0.75; ≥ 50 inliers; Δa* &gt; 6", "Scene check; redness measurement"],
    ["Database / Storage", "PostgreSQL via SQLAlchemy; SQLite fallback", "postgres:16-alpine (compose); SQLAlchemy 2.0.36", "Patients, scans, check-ins; images local or S3/MinIO"],
    ["API / Service", "FastAPI; Uvicorn; WebSocket", "0.115.6; 0.34.0; REST + <font name='Mono'>/ws/alerts</font>", "Gates, inference, persistence, alerts"],
    ["Tool / Platform", "Next.js; React; Tailwind CSS", "16.3.1; 19.2.8; 4", "Clinician dashboard"],
    ["Tool / Platform", "pytest; uv; Docker; GitHub Actions", "pytest 8.3.4; Docker 29.1.2", "Tests, environments, infrastructure, CI"]],
    [82, 130, 150, 126])
H2("4.1 Algorithm / Model Description")
shaded(["<b>Attention U-Net</b> — a 4-stage encoder–decoder whose skip connections pass through soft attention gates, so background (sheet creases, skin glare) is down-weighted before concatenation. Veins cover only about 0.6 % of a frame, so BCE alone is minimised by predicting nothing; the loss is therefore weighted BCE (<i>pos_weight</i> estimated from the masks, weight 0.3) plus soft Dice (0.7). A first run collapsed to all-background until both measures were added. Trained with AdamW (lr 1e-4, cosine decay) for 19 epochs; the epoch-16 checkpoint had the best validation Dice (0.475) and is deployed.",
        "<b>Frangi vesselness and CLAHE</b> — a hand-engineered “tube-likeness” channel and local contrast enhancement give the network the prior that veins are dark elongated ridges. scikit-image's <i>black_ridges</i> flag is a no-op in 0.25, so polarity is enforced explicitly against a local background. The network is trained with this preprocessing, so inference uses it unchanged.",
        "<b>Geometric site selector</b> — deterministic and inspectable: a clinician can see why a point was chosen (width, straightness, margin). It refuses when no segment is wide, long and straight enough. The angle is a documented heuristic, not a depth measurement.",
        "<b>YOLOv11n</b> — a small single-stage detector that runs in-process at bedside latency. No public cannula dataset exists, so it was trained on synthetic composites; the gate therefore has three states and refuses when unsure.",
        "<b>ORB + RANSAC</b> — checks that a follow-up shows the same scene as the baseline and supplies the homography used to align them; a different scene gives ≤ 11 inliers, so a threshold of 50 separates the cases. <b>CIELAB Δa*</b> — a* isolates the green–red axis, so a shadow or bruise (which mostly moves L*) is less likely to read as redness, and differences within one photo, or against an aligned and colour-matched baseline, cancel exposure and skin tone. The first measurement (difference of two in-photo references) under-measured spreading redness (r = 0.59 against injected truth) and was replaced by the baseline-anchored measurement (r = 0.995)."])
H2("4.2 Key Parameters / Hyperparameters")
table(["Parameter", "Value", "Reason / Role"], [
    ["VEIN_MASK_THRESHOLD", "0.90", "Chosen on validation only; removes speckle and halo while keeping recall 0.759. 0.95 gives higher Dice (0.592) but loses 6 pts of recall, so was not adopted."],
    ["VEIN_MIN_COMPONENT_PX", "60 px", "Removes isolated false-positive blobs smaller than a credible vein."],
    ["CLAHE clip / tile", "2.0 / 8×8", "Local contrast without amplifying noise; fixed because training used it."],
    ["Frangi σ, β", "1,3,5,7,9; 0.5", "Scales cover thin to wide veins; dark-ridge polarity enforced."],
    ["LINEARITY_R2 / WINDOW", "0.95 / 9 px", "Candidate runs must be straight and accessible."],
    ["MIN width / length", "3.0 px / 15 px", "A narrower or shorter run is not a credible target at 512×512."],
    ["Needle angle range", "15°–30°", "Shallow for wide/high-contrast (superficial) veins, steeper otherwise (heuristic)."],
    ["CANNULA_CONF_THRESHOLD", "0.50 (0.25 floor)", "Verified ≥ 0.50; 0.25–0.50 unable to verify; below 0.25 no cannula."],
    ["SCENE_MIN_INLIERS", "50", "Same scene 881–1,865 inliers, unrelated ≤ 11."],
    ["RED_DELTA_A", "6.0", "Δa* above the 2–3 unit just-noticeable skin colour difference; counts a pixel as reddened."],
    ["Ring (hub half-sizes)", "0.75 – 5.0", "Skin just outside the hub up to ≈ 2.5 hub widths."],
    ["Alert rules", "&gt; 0.6 · &gt; +0.15 · 3 rises", "Taken from the project plan; not fitted to outcome data."],
    ["Quality gate", "256 px · 20 · 40 · 25 %", "Min short side · Laplacian variance · brightness · max overexposed."]],
    [120, 80, 288], mono_cols=(0,))

# ============================================================ 5
H1("5. Dataset / Database Details")
NOTE("Complete this section only if the project uses a dataset, database, knowledge base, logs, sensor data or any other persistent data source.")
table(["Item", "Details"], [
    ["Dataset / Database Name", "CUBITAL NIR forearm dataset (vein module); synthetic cannula composites (YOLO gate); AZH and Medetec wound photos (background skin images and negatives); application database (patients, vein_scans, site_checkins)"],
    ["Source", "CUBITAL: Salcedo et al., github.com/EdwinTSalcedo/CUBITAL (public). AZH wound dataset and Medetec gallery: public downloads (scripts/download_datasets.py). Cannula composites and demo sequences: generated internally."],
    ["Size / Number of Records", "CUBITAL: 2,016 NIR images (512×512) from 1,008 subjects; 285 validation and 288 test images contain an annotated vein. AZH: 930 downloaded (830 used); Medetec: scraped galleries. Cannula set: 130 images (90 train / 15 val / 25 test)."],
    ["Features / Attributes", "CUBITAL 3-class masks (0 background, 1 arm, 2 vein; the vein class is used); YOLO hub boxes; database fields listed in Figure 6."],
    ["Train / Validation / Test Split", "CUBITAL, subject-level (seed 42): 1,412 / 302 / 302 images = 706 / 151 / 151 subjects; 0 subjects and 0 identical images shared between splits."],
    ["Data Format", "PNG/JPEG images, PNG masks, YOLO .txt labels, SQL tables, JSON results"],
    ["Pre-processing", "Subject-level split; resize to 512×512; CLAHE + bilateral; Frangi channel; augmentation for training; images downscaled to ≤ 1024 px for site checks"],
    ["Privacy / Ethics Considerations", "Public, de-identified research images only; demo patients are synthetic and labelled DEMO-; simulated site images carry an embedded IntelliCannula-Simulated marker. The prototype has no authentication and must not be used with identifiable patient data."]],
    [118, 370], bold_first=True)
H2("5.1 Data Preparation / Database Schema")
shaded(["<b>Pipeline.</b> <font name='Mono'>vein_module/src/data_loader.py --prepare</font> splits CUBITAL by subject (each person has two arm images, so a file-level split would leak), extracts the vein class (<font name='Mono'>mask == 2</font>) and writes a split manifest; <font name='Mono'>scripts/data_report.py</font> regenerates the data report. Vein pixels are about 0.6 % of a frame, which motivates the weighted BCE + Dice loss.",
        "<b>Schema.</b> <font name='Mono'>patients</font> (name, unique MRN) has many <font name='Mono'>vein_scans</font> (image and mask URLs, insertion point, angle, width, confidence, reason, modality fields, model version, image SHA-256) and many <font name='Mono'>site_checkins</font> (image URL, hub box, risk score, change, alert flag and reason, trend, registration flag, model version, SHA-256). Insertion fields are nullable on purpose: a refusal with a reason is a valid result. Audit records are structured JSON log lines (see <font name='Mono'>docs/audit_logging.md</font>), not a table."])
figure("fig6_dataset_schema.png", "Figure 6. Dataset sample / database schema / data flow: CUBITAL NIR training samples and the relational schema", max_h=360)

# ============================================================ 6
H1("6. Implementation Screenshots")
NOTE("Provide clear screenshots showing the completed implementation. Screenshots should demonstrate the major features and outputs, not only the source code editor.")
figure("fig7_dashboard.png", "Figure 7. Clinician dashboard home page (live capture, http://localhost:3000): advisory banner, model panel and patient directory with alert badges", max_h=400)
shaded(["Shows the running Next.js dashboard connected to the API (green API indicator). The model panel is read live from <font name='Mono'>/models</font> and reports the vein model's test Dice 0.567, the cannula detector's synthetic-only training and the unvalidated status of redness measurement. Patient cards show check-in counts, latest score, trend, alert and “site check due” badges."], bodyi)
figure("fig8_patient.png", "Figure 8. Patient monitoring page (live capture, patient #22): NIR-only upload notice, site capture, redness-over-time chart with the 0.6 alert line and history table", max_h=440)
shaded(["Shows the end-to-end flow after the recorded run: three check-ins at 0.11, 0.21 and 0.82, a “worsening” trend, and an alert on the third (+0.61 change versus previous). The NIR panel states that phone photos are rejected and that no live NIR device is registered; verified CUBITAL frames are available for upload."], bodyi)
figure("fig9_api_docs.png", "Figure 9. Interactive API documentation (live capture, http://localhost:8001/docs): patients, vein, infection, history and meta endpoints", max_h=330)
shaded(["Shows the REST surface used by the dashboard and the demo script, generated from the FastAPI application, confirming the back end is running with all routes registered."], bodyi)
figure("fig10_demo_run.png", "Figure 10. End-to-end demo run (<font name='Mono'>scripts/run_demo.py</font>) against the live API: six steps, all passing, including both refusals", max_h=250)
shaded(["Shows the two modules and their gates executing against the running system: an NIR frame is accepted and segmented (322 ms); an RGB photo is refused for the vein scan (HTTP 422); a simulated baseline is recorded; a mild follow-up is compared with it; the spreading follow-up raises an alert; a no-cannula photo is refused (HTTP 422) and creates no check-in. Overlay outputs for the same run are in Figures 3–5."], bodyi)

# ============================================================ 7
H1("7. Experimental Setup")
H2("7.1 Hardware Configuration")
table(["Component", "Configuration"], [
    ["Processor / CPU", "Apple M3, 8 cores"], ["Memory / RAM", "16 GB unified memory"],
    ["GPU / Accelerator", "Apple M3 integrated GPU (10 cores) via PyTorch MPS"], ["Storage", "460 GB internal SSD"],
    ["Other Hardware", "None attached. No live NIR imager is connected to this prototype, so the vein workflow runs on verified CUBITAL frames; site photos are RGB files/phone uploads."]], [150, 338], bold_first=True)
H2("7.2 Software Environment")
table(["Component", "Version / Details"], [
    ["Operating System", "macOS 26.3.1 (Apple Silicon)"], ["Programming Language", "Python 3.12.8; TypeScript 5 on Node v24.14.1"],
    ["Frameworks / Libraries", "PyTorch 2.14.0 (MPS), OpenCV 5.0.0, scikit-image 0.25.0, SciPy 1.18.1, NumPy 2.1.3, Ultralytics 8.4.123, FastAPI 0.115.6, SQLAlchemy 2.0.36, Next.js 16.3.1, React 19.2.8"],
    ["Database / Server", "SQLite fallback for the recorded runs (the Docker daemon was not running, so the API fell back automatically); PostgreSQL 16 (postgres:16-alpine) is the configured primary via docker-compose. Uvicorn 0.34.0."],
    ["IDE / Development Tools", "Visual Studio Code, Claude Code, uv, pytest, Git/GitHub Actions"],
    ["Deployment Environment", "Local: native Python (uv) for API and ML so MPS stays available; Docker for infrastructure; optional containerised stack (<font name='Mono'>docker compose --profile full</font>)"]], [150, 338], bold_first=True)
H2("7.3 Experimental / Evaluation Protocol")
shaded(["<b>Vein segmentation.</b> Subject-level split (seed 42). Training: AdamW lr 1e-4 with cosine decay, 19 epochs, best-validation checkpoint (epoch 16). Post-processing (threshold, minimum component size, TTA, hysteresis) was chosen on the validation split only (285 images with a vein). The test split (288 images with a vein) was evaluated once with the frozen configuration; error analysis on test is analysis only and changed nothing. Per-image Dice distribution, error taxonomy (using the arm class) and split-integrity checks were recorded.",
        "<b>Cannula gate.</b> Synthetic val/test for the threshold; false acceptance on 400 real images without a cannula (200 CUBITAL, 100 AZH, 100 Medetec); individual real IV photos. No labelled real cannula set exists, so real-world false rejection is not estimated.",
        "<b>Site monitoring.</b> Controlled validation (<font name='Mono'>scripts/validate_site_monitoring.py</font>) on real limb photos with known injected redness: +3 … +24 a* units, two spreads, skin only; 48 follow-ups on the 4 usable backgrounds plus lighting, white-balance, JPEG and phone-movement “nuisance” photos with true change 0. The measurement is the real API code path. Ground truth is the share of baseline-ring skin whose a* rose by ≥ 6.",
        "<b>Latency</b> was measured on Apple MPS, warm and repeated, from image-in to result-out. <b>Tests</b>: pytest suites run on 29 Oct 2026 (Appendix C). All simulated images are labelled as such and none is presented as clinical evidence."])

# ============================================================ 8
H1("8. Testing and Test Cases")
H2("8.1 Testing Strategy")
shaded(["<b>Unit tests</b> (192 tests, <font name='Mono'>vein_module/tests</font> 91 and <font name='Mono'>infection_module/tests</font> 101): data loaders and split integrity, preprocessing, model shape and parameter count, training loop, evaluation metrics, site selector, cannula detector and pipeline, registration, erythema and temporal tracker. <b>API / integration tests</b> (68 tests, <font name='Mono'>backend/tests</font>): REST contract, modality gate (11 cases: phone EXIF, plain, greyscale, “NIR” file name, device without or with wrong token, corrupt and oversized files, registered device, dataset frame), detection and quality gates, site monitoring (baseline, follow-up, different scene, blur, weak detection, thresholds, UTC timestamps, patient isolation). <b>System test</b>: <font name='Mono'>scripts/run_demo.py</font> drives six steps through the running API and asserts each outcome. <b>Model validation</b> and <b>performance testing</b> are reported in Section 9. Frontend: <font name='Mono'>tests/timeline.test.ts</font>, type-check and lint (see README)."])
H2("8.2 Test Case Summary")
table(["TC ID", "Module / Feature", "Test Input / Condition", "Expected Output", "Actual Output", "Status"], [
    ["TC01", "Modality gate", "Phone-style RGB photo (demo file 4_rgb_photo…) sent to vein-scan", "HTTP 422; model not run", "HTTP 422 “NIR vein scan required … cannot be verified” in 7 ms", "Pass"],
    ["TC02", "Modality gate / M1", "CUBITAL test-split NIR frame (1_vein_scan_arm.png)", "HTTP 201; verified; site or refusal with reason", "201; verified (cubital_dataset, test split); site (269, 238), 4.24 px, 25.1°, Grade C, conf 0.50; 191–322 ms", "Pass"],
    ["TC03", "Modality gate", "Plain / greyscale / “NIR”-named image; device ID without or with wrong token", "HTTP 422 unverified; model never called", "Rejected in automated tests (test_modality.py)", "Pass"],
    ["TC04", "Modality gate", "Registered device with matching token; camera-EXIF photo", "Device accepted; EXIF photo rejected as RGB", "As expected in automated tests", "Pass"],
    ["TC05", "Quality gate", "Blurred, dark and overexposed site photos", "Refused before scoring", "Refused (test_detection_gates.py, test_site_monitoring.py)", "Pass"],
    ["TC06", "Cannula gate", "Landscape photo with no cannula", "HTTP 422; nothing stored", "422 “No cannula detected … retake” (82 ms); no check-in created", "Pass"],
    ["TC07", "Cannula gate / M2", "Simulated baseline photo with hub", "Hub verified; baseline recorded", "Confidence 0.71, 201, “Baseline recorded”, reddened 5 %, score 0.11", "Pass"],
    ["TC08", "Scene check", "Follow-up of an unrelated scene", "Not compared (&lt; 50 inliers)", "Refused as a different scene (test_site_monitoring.py)", "Pass"],
    ["TC09", "M3 / compare", "Mild follow-up (day 2)", "Compared with baseline; no alert", "1,895 inliers; “Visible change detected”, reddened 10 %, change +54 %, score 0.21, no alert", "Pass"],
    ["TC10", "Alert engine / M4", "Spreading follow-up (day 3)", "Alert raised and shown", "1,866 inliers; “Review recommended”, reddened 41 %, change +84 %, score 0.82 (+0.61), alert", "Pass"]],
    [32, 58, 104, 82, 150, 44], centre_cols=(0, 5), fs=7)
H2("8.3 Defects Identified and Resolved")
table(["Issue ID", "Issue / Defect", "Resolution", "Final Status"], [
    ["I01", "Vein mask was blobs and speckle; 74 % of marked pixels were not vein (training stopped at epoch 6 of 25; fixed 0.5 threshold)", "Resumed to epoch 16; threshold 0.9 and 60 px filter tuned on validation. Test Dice 0.370 → 0.567", "Resolved"],
    ["I02", "A clear 13.8 px vein was graded “Not Recommended” (grade used the selector's separation margin)", "Grade from vein width; refusal maps to “Not Recommended”", "Resolved"],
    ["I03", "A landscape photo scored infection risk 0.999 (no cannula check; image centre scored)", "Strict cannula gate; no centre fallback", "Resolved"],
    ["I04", "Every site photo scored ≈ 1.00 (LightGBM trained on chronic wound vs skin proxy labels)", "Replaced by measured peri-cannula redness versus the patient's baseline", "Resolved"],
    ["I05", "RGB phone photos were accepted for vein scanning (no modality check)", "NIR provenance gate (SHA-256 registry / device token; EXIF rejected); model never called on rejection", "Resolved"],
    ["I06", "Follow-up of a different scene was compared as the same site", "ORB/RANSAC scene check; &lt; 50 inliers refused", "Resolved"],
    ["I07", "Site check took 1.7–2.8 s (detector started in a new subprocess per request)", "In-process cached detector: 42 ms", "Resolved"],
    ["I08", "Vein forward pass 597 ms on CPU", "Apple MPS: 77 ms with identical masks (40/40 test images, Dice 1.0000)", "Resolved"],
    ["I09", "Change since baseline under-measured (r = 0.59, mean error 21.5 pts)", "Pixel-wise comparison with the aligned, colour-matched baseline: r = 0.995, mean error 3.6 pts", "Resolved"],
    ["I10", "Most-inflamed follow-ups refused (13 of 48): detector confidence drops on red skin", "Weak detection accepted on a follow-up that matches a verified baseline; 0 of 48 refused", "Resolved"],
    ["I11", "Chart times 5.5 h early; dashboard showed stale hard-coded metrics", "UTC normalisation with regression tests; live model panel from /models", "Resolved"]],
    [38, 200, 200, 50], centre_cols=(0, 3))

# ============================================================ 9
H1("9. Results and Performance Metrics")
H2("9.1 Final Results")
shaded(["All four objectives were achieved and demonstrated end to end on the running system. <b>O1:</b> the vein model is never called on an RGB or unverified image (11 gate tests; phone photo refused in the live run). <b>O2:</b> test Dice 0.567 (median 0.610) against the 0.55 target, up from 0.370 at the first checkpoint; no false positives off the arm beyond 0.3 % of false-positive pixels. <b>O3:</b> the selector returns a point, width, angle and grade, or refuses with a reason. <b>O4:</b> the cannula gate rejects 98–99 % of real non-cannula images; scenes are matched with a wide margin (881–1,865 inliers versus ≤ 11 against a threshold of 50); the baseline-anchored change measurement tracks injected redness with r = 0.995 and raised 0 false alarms on lighting, white-balance, compression and movement photos. These results show that the system <i>measures</i> redness change correctly under controlled conditions; they do not show that redness predicts phlebitis, which needs nurse-assessed IV-site photographs."])
H2("9.2 Performance Metrics")
table(["Metric", "Result", "Target / Benchmark", "Interpretation"], [
    ["Vein Dice (test, 288 vein images)", "0.567 (median 0.610; P10 0.350, P90 0.741)", "≥ 0.55 (O2); first checkpoint 0.370", "Target met. 17.7 % of images score below 0.4."],
    ["Vein IoU", "0.413", "First checkpoint 0.239", "+73 % relative."],
    ["Vein precision / recall", "0.495 / 0.758", "First checkpoint 0.262 / 0.814", "Precision nearly doubled; recall traded for fewer false positives."],
    ["False positives off the arm", "0.3 % of FP pixels", "n/a", "47 % within 2 px of a true vein, 53 % on the arm elsewhere."],
    ["Cannula gate false acceptance (real, no cannula)", "1 % CUBITAL, 1 % AZH, 2 % Medetec (n = 200/100/100)", "—", "A false acceptance on a follow-up is still stopped by the scene check."],
    ["Cannula gate on synthetic test", "0 % false acceptance, 5 % false rejection (1/19)", "—", "Synthetic only. One real IV-dressing photo accepted (0.79); one real hand photo missed (0.19)."],
    ["YOLOv11n mAP@50 (synthetic val)", "0.995", "—", "Measured on synthetic composites; not evidence of real-world accuracy."],
    ["Scene check inliers", "881–1,865 same scene; ≤ 11 unrelated", "Threshold 50", "Wide margin on simulated moves (rotation 8°, re-save)."],
    ["Change measurement vs injected redness", "r = 0.995; MAE 3.6 pts; 16/16 detected, 8/8 not flagged", "Original method r = 0.59, MAE 21.5", "With a white card in frame: r = 0.996, MAE 3.0, 24/24 and 12/12. 0 of 27 nuisance photos falsely flagged."],
    ["Vein model forward pass (MPS)", "77 ms", "597 ms on CPU", "Identical masks on 40/40 test images."],
    ["Vein request, warm (MPS)", "185 ms (live run 191–322 ms)", "3.4 s first request", "Includes model load and NIR-registry build on the first call."],
    ["Site check request", "158–289 ms (live run)", "1.7–2.8 s before caching", "Detector 42 ms cached; whole check includes scene and redness."],
    ["Automated tests", "260 passed (192 + 68), 0 failed", "—", "21.83 s + 4.48 s on 29 Oct 2026."]],
    [112, 132, 108, 136])
H2("9.3 Result Figures / Graphs")
figure("fig11_results.png", "Figure 11. (a) Validation sweep of inference-time settings; (b) measured versus injected redness (r = 0.995) across lighting changes", max_h=250)
shaded(["(a) The deployed setting (threshold 0.90, 60 px filter) balances precision and recall; threshold 0.95 has the highest Dice but misses more vein, and flip-TTA adds 0.005 Dice for twice the latency, so neither was adopted. (b) Each point is one follow-up of the controlled validation: measured change tracks the injected change closely and slightly under-measures (worst case 48 % true vs 35 % measured), i.e. it errs toward missing a little change rather than inventing it."], bodyi)

# ============================================================ 10
H1("10. Comparison and Analysis")
NOTE("Use this section when a baseline, existing method, previous version or competing approach is available.")
table(["Method / System", "Vein Dice", "Vein precision / recall", "Change vs injected truth", "Observation"], [
    ["First checkpoint (epoch 6, threshold 0.5; in-photo reference)", "0.370", "0.262 / 0.814", "r = 0.59, MAE 21.5 pts", "Speckled masks, markers on the bedsheet, change under-measured, no gates."],
    ["Threshold 0.95 (validation only, not adopted)", "0.592 (val)", "0.557 / 0.701", "—", "Higher Dice from discarding uncertain pixels; misses ~6 pts more vein."],
    ["<b>Implemented system (epoch 16, threshold 0.90, baseline-anchored change)</b>", "<b>0.567</b>", "<b>0.495 / 0.758</b>", "<b>r = 0.995, MAE 3.6 pts</b>", "Gated, auditable, measured; limitations in Section 11."]],
    [128, 52, 78, 90, 140], centre_cols=(1, 2))
H2("10.1 Analysis of Results")
shaded(["<b>Strengths.</b> Most of the gain came from resuming training to epoch 16 and from threshold 0.90 with a 60 px filter; false positives almost disappeared from the background (0.3 % off-arm). The largest improvement in the site module came from changing <i>what is measured</i>: comparing pixels with an aligned, colour-matched baseline instead of subtracting two in-photo references, because spreading redness also reddens the far skin used as the reference and cancels itself. Gates (NIR provenance, quality, cannula, scene) convert silent failures into explicit refusals.",
        "<b>Trade-offs and failure cases.</b> Recall was traded for precision (0.814 → 0.758). Thin veins (&lt; 4 px, 22.6 % of vein pixels) have recall 0.70 against 0.76 for wider ones, and veins are often fragmented (median 1.5 predicted pieces per annotated vein). High-contrast frames score lowest (Dice 0.536 versus 0.591 for low contrast); in the worst case (Dice 0.02) the annotation omits a visible vein, so some “false positives” are likely unannotated veins (a hypothesis that needs re-annotation to test). The cannula detector trained on synthetic images missed the hub on 8 of 12 validation backgrounds, including the real IV-hand photo; only 4 backgrounds could be used. Without a white or grey object away from the site, a lighting change cannot be told from redness, so such comparisons are reported as “unable to compare reliably” instead of guessed (24 of 48 follow-ups deferred as photographed; 12 of 48 with a white card).",
        "<b>Practical significance.</b> The vein module is a decision aid that shows its reasoning and refuses when unsure; agreement with practitioner-marked insertion points has not yet been measured. The site module shows that redness change can be measured reproducibly on a phone photo; whether that predicts phlebitis is an open clinical question."])

# ============================================================ 11
H1("11. Limitations")
shaded(["<b>1. Hardware:</b> vein scanning needs NIR imaging (~850 nm); no NIR device is attached, so the workflow runs on verified CUBITAL frames and cannot use an ordinary phone camera.",
        "<b>2. Population:</b> CUBITAL is 1,008 subjects aged 5–22 (median 11), 97 % recorded as slim; transfer to elderly, obese or dehydrated adults, where veins are hardest to find, is untested.",
        "<b>3. Site selection is unvalidated against practice:</b> agreement with nurse-marked insertion points was not computed; the needle angle is a width/contrast heuristic, not a measured depth; in the live run the selector had a single candidate (confidence 0.50, Grade C).",
        "<b>4. Segmentation quality:</b> test Dice 0.567 is moderate; 17.7 % of test images score below 0.4; thin veins and fragmented veins are the main misses.",
        "<b>5. Cannula detector:</b> trained on 90 synthetic composites; real-world accuracy is unmeasured (one real hand photo was missed); it may degrade with tape, blood staining or opaque dressings.",
        "<b>6. No longitudinal clinical data:</b> site sequences are simulated; alert thresholds (0.6, +0.15, three rises) come from the project plan and their false-alarm and miss rates are unmeasured; redness change cannot confirm or exclude infection.",
        "<b>7. Validation scope:</b> controlled validation used 4 mostly light-skin backgrounds, smooth injected redness and a white/grey reference; darker skin and streaking phlebitis are not covered.",
        "<b>8. Deployment:</b> the prototype has no authentication or clinical safeguards beyond the advisory design; synthetic or de-identified data only; research prototype, not a medical device."])

# ============================================================ 12
H1("12. Conclusion")
shaded(["IntelliCannula implements both modules of the proposed system and integrates them behind one API and dashboard. The pre-cannulation module verifies NIR provenance, segments veins with an Attention U-Net (test Dice 0.567, median 0.610; 77 ms forward pass on MPS) and selects an insertion site with width, angle and grade, refusing when unsure. The post-cannulation module gates photo quality, verifies the cannula hub, checks scene continuity (≥ 50 inliers) and measures baseline-anchored CIELAB redness change (r = 0.995 against injected truth) with advisory alerts. 260 automated tests pass and the full workflow runs end to end against the live system, including the refusal paths.",
        "All four project objectives are met and the Review 3 implementation is complete. The contribution is a working, auditable decision-support pipeline whose limits are stated; clinical validation (adult cohorts, nurse-marked insertion points, longitudinal IV-site photographs with phlebitis grading, real NIR hardware) is the next step and is not claimed here."])

# ============================================================ 13
H1("13. Instructions for Executing the Project")
H2("13.1 Prerequisites")
shaded(["macOS (Apple Silicon recommended for MPS) or Linux; Python 3.12 with <font name='Mono'>uv</font>; Node.js 20 or newer (tested on v24); Docker (optional, PostgreSQL); <font name='Mono'>brew install libomp</font> on macOS (LightGBM, imported by the legacy risk module). Trained checkpoints (<font name='Mono'>vein_module/models/attention_unet_best.pt</font>, <font name='Mono'>infection_module/models/cannula_yolo11.pt</font>) and <font name='Mono'>data/demo/</font> are supplied in the submission folder (git-ignored). Optional environment variables: <font name='Mono'>DATABASE_URL</font> (SQLite is used automatically when PostgreSQL is unreachable), <font name='Mono'>NIR_DEVICES</font>, <font name='Mono'>ML_DEVICE</font>."])
H2("13.2 Installation / Setup Steps")
steps("s", ["<font name='Mono'>cd intellicannula &amp;&amp; cp .env.example .env</font>",
            "<font name='Mono'>uv venv infection_module/.venv --python 3.12</font>",
            "<font name='Mono'>uv pip install --python infection_module/.venv -r infection_module/requirements.txt -r vein_module/requirements.txt</font>",
            "<font name='Mono'>uv pip install --python infection_module/.venv -r backend/requirements.txt</font>",
            "<font name='Mono'>cd frontend &amp;&amp; npm install</font>  (optional: <font name='Mono'>docker compose up -d</font> for PostgreSQL)"])
H2("13.3 Execution Steps")
steps("e", ["<font name='Mono'>cd backend &amp;&amp; ../infection_module/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001</font>  →  http://localhost:8001/health",
            "<font name='Mono'>cd frontend &amp;&amp; npm run dev -- -H 0.0.0.0 -p 3000</font>  →  http://localhost:3000",
            "<font name='Mono'>infection_module/.venv/bin/python scripts/build_demo_set.py</font>  (writes data/demo/)",
            "<font name='Mono'>infection_module/.venv/bin/python scripts/run_demo.py</font>  (API running; prints PASS for six steps)",
            "<font name='Mono'>infection_module/.venv/bin/python -m pytest -q</font>  and  <font name='Mono'>cd backend &amp;&amp; ../infection_module/.venv/bin/python -m pytest -q</font>"])
H2("13.4 Sample Input and Expected Output")
table(["Item", "Example / Description"], [
    ["Sample Input", "data/demo/1_vein_scan_arm.png (CUBITAL test-split NIR frame); data/demo/SIMULATED_day1…3 site photos"],
    ["Command / Action", "<font name='Mono'>python scripts/run_demo.py</font>, or upload the files from the patient page at http://localhost:3000/patients/&lt;id&gt;, or <font name='Mono'>POST /patients/{id}/vein-scan</font> with the file"],
    ["Expected Output", "Vein scan 201: verified, site (269, 238), width 4.24 px, 25.1°, Grade C; RGB photo 422; site checks 201 with scores 0.11 → 0.21 → 0.82 and an alert on the third; no-cannula photo 422; “all steps passed”"]],
    [110, 378], bold_first=True)
H2("13.5 Troubleshooting")
table(["Problem", "Possible Cause", "Resolution"], [
    ["HTTP 422 “NIR vein scan required”", "RGB/phone photo or an unverified image sent to vein-scan", "Upload a verified CUBITAL NIR frame, or register a device with <font name='Mono'>NIR_DEVICES</font> and send both headers"],
    ["HTTP 503 naming a missing package", "API started from an environment without torch/OpenCV", "Start Uvicorn with <font name='Mono'>infection_module/.venv/bin/python</font>"],
    ["HTTP 422 “No cannula detected / unable to verify”", "Hub not clearly in frame, or detector below 0.50", "Retake with the hub and dressing centred, ~20 cm away, with a white sheet in frame"],
    ["“Unable to compare reliably”", "No white/grey reference away from the site, or different scene (&lt; 50 inliers)", "Include a white sheet or paper; photograph the same site from a similar view"],
    ["Database connection warning", "PostgreSQL not running", "Automatic SQLite fallback; or <font name='Mono'>docker compose up -d</font>"]],
    [140, 168, 180])

# ============================================================ appendices
H1("Appendix A. Source Code Structure")
NOTE("Provide the folder/file structure and a short description of important files.")
table(["Path / File", "Purpose"], [
    ["intellicannula/", "Project root"],
    ["backend/app/main.py, api/, core/, models/, schemas/", "FastAPI app: routes (patients, vein, infection, history, alerts), settings, DB engine, ML bridge, audit log, ORM models"],
    ["backend/app/core/modality.py", "NIR provenance gate (SHA-256 registry, device token, EXIF rejection)"],
    ["backend/app/core/ml.py", "Model cache, quality gate, vein and site-check pipelines"],
    ["vein_module/src/", "data_loader, preprocessing (CLAHE, Frangi), model (Attention U-Net), train, evaluate, postprocess, site_selector, error_analysis"],
    ["infection_module/src/", "cannula_detector/, erythema.py, registration.py, temporal_tracker.py (alert rules), legacy risk_model.py"],
    ["frontend/app, components, lib", "Next.js dashboard: patient directory, patient page, charts, API client"],
    ["data/", "raw/, processed/ and demo/ inputs and split manifests (git-ignored; supplied separately)"],
    ["vein_module/models/, infection_module/models/", "Trained checkpoints (Attention U-Net, YOLOv11n cannula detector)"],
    ["backend/tests, vein_module/tests, infection_module/tests, frontend/tests", "Automated tests (260 Python tests)"],
    ["scripts/", "download_datasets, build_demo_set, run_demo, validate_site_monitoring, benchmark_latency, data_report, full_system_eval, report generators"],
    ["docs/", "Measured reports: vein_error_analysis, REVIEW2_RESULTS, site_monitoring_validation, cannula_gate_eval.json, audit_logging, data_report"],
    ["docker-compose.yml, .env.example, .github/workflows/", "Infrastructure, configuration template, CI"],
    ["requirements.txt (backend/, vein_module/, infection_module/)", "Dependencies"],
    ["README.md", "Setup and execution guide"]], [190, 298], mono_cols=(0,))
H1("Appendix B. Dependencies and Configuration")
table(["Dependency / Tool", "Version", "Installation / Configuration Note"], [
    ["Python / uv", "3.12.8", "<font name='Mono'>uv venv … --python 3.12</font>"],
    ["torch (+ MPS)", "2.14.0", "In infection_module/.venv; <font name='Mono'>ML_DEVICE=auto</font> selects MPS/CUDA/CPU"],
    ["opencv / scikit-image / scipy / numpy", "5.0.0 / 0.25.0 / 1.18.1 / 2.1.3", "Preprocessing, vesselness, registration, colour"],
    ["ultralytics", "8.4.123", "YOLOv11n inference and training"],
    ["fastapi / uvicorn / sqlalchemy", "0.115.6 / 0.34.0 / 2.0.36", "<font name='Mono'>backend/requirements.txt</font> (also psycopg2, boto3, pydantic 2.10.4, websockets 14.1)"],
    ["Next.js / React / Tailwind", "16.3.1 / 19.2.8 / 4", "<font name='Mono'>cd frontend &amp;&amp; npm install</font>"],
    ["PostgreSQL", "16 (postgres:16-alpine)", "<font name='Mono'>docker compose up -d</font>; <font name='Mono'>DATABASE_URL</font> in .env"],
    ["Thresholds", "VEIN_MASK_THRESHOLD 0.9; VEIN_MIN_COMPONENT_PX 60; CANNULA_CONF_THRESHOLD 0.5; SCENE_MIN_INLIERS 50", "Environment variables, <font name='Mono'>backend/app/core/config.py</font>"]],
    [140, 150, 198])
H1("Appendix C. Additional Evidence")
shaded(["<b>C1 – Error analysis of the deployed vein model</b> (median test frame; <font name='Mono'>docs/vein_error_analysis.md</font>). <b>C2 – Controlled site-change validation</b> examples (<font name='Mono'>docs/site_monitoring_validation.md</font>). <b>C3 – Automated test run</b> of 29 Oct 2026. Further raw evidence: <font name='Mono'>docs/vein_postprocess_experiments.json</font>, <font name='Mono'>docs/vein_error_analysis.json</font>, <font name='Mono'>docs/cannula_gate_eval.json</font>, <font name='Mono'>docs/site_monitoring_validation.json</font>, <font name='Mono'>docs/latency_benchmark.json</font>."])
figure("figC1_error_breakdown.png", "Figure C1. Vein segmentation error breakdown on the median test frame", max_h=200)
figure("figC2_site_validation.png", "Figure C2. Controlled site-change validation: baseline (left) and change since baseline at +9 and +18 a* (magenta = redder than baseline; pre-existing wounds are correctly not counted). Labels give true vs measured change.", max_w=330, max_h=250)
figure("figC3_pytest.png", "Figure C3. Automated test results (29 Oct 2026)", max_h=130)

# ============================================================ declaration
story.append(P("Student / Team Declaration", S("dh", fontName="Body-B", fontSize=15, leading=19, textColor=colors.HexColor("#1B365D"), spaceBefore=12, spaceAfter=6, keepWithNext=1)))
story.append(P("We confirm that the implementation documented in this report represents the completed Review 3 version of the project and that the submitted code corresponds to the demonstrated system.", S("dp", fontSize=9.5, leading=13, textColor=colors.HexColor("#222222"), spaceAfter=8)))
dt = Table([[P("Name", hdr), P("Registration No.", hdr), P("Signature", hdr), P("Date", hdr)],
            [P("Sastika M", cell), P("23MIA1047", cell), "", P("29-10-2026", cell)],
            [P("Akshaya GK", cell), P("23MIA1127", cell), "", P("29-10-2026", cell)]], colWidths=[150, 120, 120, 98], rowHeights=[None, 24, 24])
dt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("GRID", (0, 0), (-1, -1), 0.6, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
story.append(KeepTogether([dt]))

import json, os
if os.environ.get("DOC_MODEL"):
    Path(os.environ["DOC_MODEL"]).write_text(json.dumps(MODEL, ensure_ascii=False))
doc.build(story, canvasmaker=Canvas)
print("wrote", OUT)
