"""Controlled validation of site-change monitoring against KNOWN injected redness.

    cd backend && ../infection_module/.venv/bin/python ../scripts/validate_site_monitoring.py

What this does and does not show
--------------------------------
Shows: whether the deployed pipeline (quality gate -> cannula detector -> scene
check -> redness measurement, `app.core.ml.run_site_check`) *measures* redness
change correctly, by putting a known amount of redness on real limb photos and
comparing the measured change with the truth - and whether lighting, white
balance, compression or moving the phone cause false alarms.

Does not show: that measured redness predicts phlebitis or infection. That
needs real IV-site photos with nurse assessments. Every image here is
SIMULATED (procedural cannula hub, injected redness) and carries the PNG marker.

Ground truth for a follow-up = fraction of the baseline's measured skin ring
where the injected a* increase is >= 6 (the system's own redness threshold).
Measured = `change_since_baseline` returned by the API pipeline.

Backgrounds: one real IV-hand photo + 11 Medetec limb photos with visible
skin. Wounds in a background are present on every day, so they cancel in the
change measurement. The hub is placed automatically at the skin point
furthest from non-skin (no manual placement).
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.core import ml  # noqa: E402
from infection_module.src import erythema  # noqa: E402
from infection_module.src.cannula_detector.generate_dataset import (  # noqa: E402
    GAUGE_COLORS, create_cannula_hub_rgba,
)

MED = ROOT / "data/raw/medetec/slide scans"
BACKGROUNDS = [ROOT / "backend/data/uploads/patients/1/site/20261009/ac619862d2ffa641.png"] + [
    MED / p for p in [
        "extravasation-wound-images/images/extravasation-wound-a-003.jpg",
        "extravasation-wound-images/images/extravasation-wound-a-005.jpg",
        "extravasation-wound-images/images/extravasation-wound-a-006.jpg",
        "extravasation-wound-images/images/extravasation-wound-a-007.jpg",
        "extravasation-wound-images/images/extravasation-wound-b-003.jpg",
        "extravasation-wound-images/images/extravasation-wound-b-004.jpg",
        "leg-ulcer-images/images/leg-ulcers-0002.jpg",
        "leg-ulcer-images/images/leg-ulcers-0009.jpg",
        "leg-ulcer-images/images/leg-ulcers-0011.jpg",
        "burns/images/case2-burn-to-arm-1.jpg",
        "burns/images/case2-burns-to-arm-7.jpg",
    ]
]
# Backgrounds with no white/grey object away from the site (measured: neutral
# fraction below NEUTRAL_MIN_FRACTION) - re-run with a white card in frame.
CARD_BACKGROUNDS = {"extravasation-wound-a-007", "extravasation-wound-b-003"}
AMOUNTS = [3.0, 6.0, 9.0, 12.0, 18.0, 24.0]     # injected peak a* units
SPREADS = [1.0, 1.6]                            # Gaussian sigma, in hub widths
WIDTH_FRACS = [0.22, 0.18, 0.26]                # hub size tried in order until detected
OUT = ROOT / "data/demo/site_validation"
DOCS = ROOT / "docs"


def png_bytes(bgr: np.ndarray, description: str) -> bytes:
    meta = PngInfo()
    meta.add_text(ml.SIMULATED_KEY, "true")
    meta.add_text("Description", description)
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buffer, format="PNG", pnginfo=meta)
    return buffer.getvalue()


def place_hub(bg: np.ndarray, width_frac: float, gauge: dict) -> tuple[np.ndarray, tuple[int, int], float]:
    """Composite a hub at the skin point furthest from any non-skin pixel."""
    skin = erythema.skin_mask(bg).astype(np.uint8)
    dist = cv2.distanceTransform(skin, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    random.seed(7)
    rgba, (hx1, hy1, hx2, hy2) = create_cannula_hub_rgba(
        hub_width=130, hub_height=85, gauge=gauge, with_tegaderm=True, with_tape=True)
    hub = Image.fromarray(rgba)
    scale = bg.shape[1] * width_frac / (hx2 - hx1)
    hub = hub.resize((max(1, int(hub.width * scale)), max(1, int(hub.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    canvas.paste(hub, (int(cx - hub.width / 2), int(cy - hub.height / 2)), hub)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR), (int(cx), int(cy)), (hx2 - hx1) * scale


def add_white_card(bg: np.ndarray) -> np.ndarray:
    """A white reference card (paper) in the corner furthest from the skin centre,
    as the capture protocol asks. Added to the scene, so later lighting changes
    affect it exactly as they affect the skin."""
    skin = erythema.skin_mask(bg).astype(np.uint8)
    cy, cx = np.unravel_index(int(np.argmax(cv2.distanceTransform(skin, cv2.DIST_L2, 5))), skin.shape)
    h, w = bg.shape[:2]
    size = int(w * 0.14)
    x0 = 4 if cx > w / 2 else w - size - 4
    y0 = 4 if cy > h / 2 else h - size - 4
    out = bg.copy()
    rng = np.random.default_rng(0)
    out[y0:y0 + size, x0:x0 + size] = np.clip(225 + rng.normal(0, 2, (size, size, 3)), 0, 255).astype(np.uint8)
    return out


def added_a(shape, centre, amount, sigma) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return amount * np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * sigma**2))


def redden(img: np.ndarray, delta_a: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] += delta_a
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def nuisances(img: np.ndarray) -> dict[str, np.ndarray]:
    """Changes that are NOT biological: the system must not call them redness."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 6, 0.95)
    rot[:, 2] += [w * 0.03, -h * 0.02]
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return {
        "darker exposure (x0.75)": np.clip(f * 0.75, 0, 255).astype(np.uint8),
        "brighter exposure (x1.2)": np.clip(f * 1.2, 0, 255).astype(np.uint8),
        "warm white balance": np.clip(f * [0.90, 1.0, 1.10], 0, 255).astype(np.uint8),
        "cool white balance": np.clip(f * [1.10, 1.0, 0.92], 0, 255).astype(np.uint8),
        "JPEG quality 55": cv2.imdecode(jpg, 1),
        "phone moved (6 deg, shift)": cv2.warpAffine(img, rot, (w, h), borderMode=cv2.BORDER_REFLECT),
    }


def true_change(base_bgr: np.ndarray, hub: list[float], delta_a: np.ndarray,
                follow_bgr: np.ndarray) -> tuple[float, float]:
    """(realized, injected) share of the baseline's ring skin reddened by >= threshold.

    Realized is the a* change actually present in the saved image; it is lower
    than the injected amount where strongly red colours clip at the RGB gamut.
    Realized is the primary truth - it is what any camera would record.
    """
    result = erythema.measure(base_bgr, hub)
    size = (result["_bgr"].shape[1], result["_bgr"].shape[0])
    ring_skin = result["_ring"] & erythema.skin_mask(result["_bgr"])

    def a_star(img):
        small = cv2.resize(img, size, interpolation=cv2.INTER_AREA) if img.shape[1] != size[0] else img
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)

    realized = a_star(follow_bgr) - a_star(base_bgr)
    injected = cv2.resize(delta_a, size) if delta_a.shape[1] != size[0] else delta_a
    t = erythema.RED_DELTA_A
    return float((realized[ring_skin] >= t).mean()), float((injected[ring_skin] >= t).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nuisance_rows, skipped, examples = [], [], [], []

    runs = [(i, p, False) for i, p in enumerate(BACKGROUNDS)] + [
        (i, p, True) for i, p in enumerate(BACKGROUNDS) if p.stem in CARD_BACKGROUNDS]
    for index, path, card in runs:
        bg = cv2.imread(str(path))
        if bg is None:
            skipped.append((path.name, "missing"))
            continue
        if card:
            bg = add_white_card(bg)
        base = None
        attempts = [(g, f) for g in GAUGE_COLORS for f in WIDTH_FRACS]
        for gauge, frac in attempts:
            day1, centre, hub_w = place_hub(bg, frac, gauge)
            day1_bytes = png_bytes(day1, "SIMULATED baseline: procedural cannula hub on a real limb photo.")
            try:
                base = ml.run_site_check(day1_bytes, 0, None)
                break
            except ValueError as exc:
                last_error = str(exc)
        if base is None:
            skipped.append((path.name, last_error[:90]))
            continue
        hub_box = ml._locate_roi(day1)[2]   # the detected hub, as the pipeline uses it

        for spread in SPREADS:
            for amount in AMOUNTS:
                # Erythema is in skin, not in tape, dressing, plastic or sheets:
                # inject only where the baseline shows skin (soft-edged).
                skin = cv2.GaussianBlur(erythema.skin_mask(day1).astype(np.float32), (7, 7), 0)
                delta = added_a(day1.shape, centre, amount, spread * hub_w) * skin
                follow = redden(day1, delta)
                truth, injected = true_change(day1, hub_box, delta, follow)
                try:
                    res = ml.run_site_check(png_bytes(follow, f"SIMULATED: +{amount} a*"), 0, day1_bytes)
                    measured, status = res["change_since_baseline"], res["comparison_status"]
                    verification = res["cannula_verification"]
                except ValueError as exc:
                    measured, status, verification = None, f"refused: {str(exc)[:60]}", None
                rows.append({"background": path.name, "card": card, "spread": spread, "amount": amount,
                             "true_change": truth, "injected_change": injected,
                             "measured_change": measured, "status": status, "verification": verification})
                if spread == 1.6 and amount in (9.0, 18.0) and measured is not None and status == "compared":
                    examples.append((path.name, amount, truth, measured,
                                     base["overlay_png_base64"], res["change_overlay_png_base64"]))
                if spread == 1.6 and amount in (9.0, 18.0):
                    day = 2 if amount == 9.0 else 3
                    (OUT / f"SIMULATED_{path.stem}{'_card' if card else ''}_day{day}_plus{int(amount)}a.png").write_bytes(
                        png_bytes(follow, f"SIMULATED day {day}: +{amount} a* redness on skin"))
        (OUT / f"SIMULATED_{path.stem}{'_card' if card else ''}_day1_baseline.png").write_bytes(day1_bytes)

        for name, img in nuisances(day1).items():
            try:
                res = ml.run_site_check(png_bytes(img, f"SIMULATED nuisance: {name}"), 0, day1_bytes)
                nuisance_rows.append({"background": path.name, "card": card, "nuisance": name,
                                      "measured_change": res["change_since_baseline"],
                                      "status": res["comparison_status"]})
            except ValueError as exc:
                nuisance_rows.append({"background": path.name, "card": card, "nuisance": name,
                                      "measured_change": None, "status": f"refused: {str(exc)[:60]}"})
        print(f"done {path.name}", flush=True)

    summarise(rows, nuisance_rows, skipped, examples)


def method_stats(rows, nuisance_rows, key: str, truth: str = "true_change") -> dict:
    """Accuracy over follow-ups the system actually compared (colour-normalised)."""
    ok = [r for r in rows if r.get(key) is not None and r["status"] == "compared"]
    t = np.array([r[truth] for r in ok])
    m = np.array([r[key] for r in ok])
    v = 0.10
    tp = int(((t >= v) & (m >= v)).sum()); fn = int(((t >= v) & (m < v)).sum())
    tn = int(((t < v) & (m < v)).sum()); fp = int(((t < v) & (m >= v)).sum())
    nok = [r for r in nuisance_rows if r.get(key) is not None and r["status"] == "compared"]
    return {
        "deferred_no_colour_reference": sum(r["status"] == "no_colour_reference" for r in rows),
        "nuisance_deferred_no_colour_reference": sum(r["status"] == "no_colour_reference" for r in nuisance_rows),
        "n": len(ok), "pearson_r": float(np.corrcoef(t, m)[0, 1]),
        "mae_points": float(np.mean(np.abs(m - t)) * 100), "bias_points": float(np.mean(m - t) * 100),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "by_amount": {str(a): {"true": float(np.mean([r[truth] for r in ok if r["amount"] == a])),
                               "measured": float(np.mean([r[key] for r in ok if r["amount"] == a]))}
                      for a in AMOUNTS if any(r["amount"] == a for r in ok)},
        "nuisance_false_visible_change": f"{sum(r[key] >= v for r in nok)}/{len(nok)}",
        "nuisance_mean_abs_points": {name: float(np.mean([abs(r[key]) for r in nok if r["nuisance"] == name]) * 100)
                                     for name in sorted({r["nuisance"] for r in nok})},
    }


def summarise(rows, nuisance_rows, skipped, examples) -> None:
    photographed = [r for r in rows if not r["card"]]
    photographed_n = [r for r in nuisance_rows if not r["card"]]
    with_card = [r for r in rows if r["card"] or r["background"].rsplit(".", 1)[0] not in CARD_BACKGROUNDS]
    with_card_n = [r for r in nuisance_rows if r["card"] or r["background"].rsplit(".", 1)[0] not in CARD_BACKGROUNDS]
    deployed = {
        "as photographed": method_stats(photographed, photographed_n, "measured_change"),
        "white card added where the scene had no white/grey reference": method_stats(with_card, with_card_n, "measured_change"),
        "as photographed, truth = injected": method_stats(photographed, photographed_n, "measured_change", "injected_change"),
        "refused_follow_ups": [r for r in rows if r["measured_change"] is None],
        "refused_nuisance": [r for r in nuisance_rows if r["measured_change"] is None],
        "cannula_verification": dict(__import__("collections").Counter(r.get("verification") for r in rows)),
    }
    (DOCS / "site_method_deployed.json").write_text(json.dumps(deployed, indent=2, default=str))
    print(json.dumps(deployed, indent=2, default=str))
    ok = [r for r in rows if r["measured_change"] is not None]
    t = np.array([r["true_change"] for r in ok])
    m = np.array([r["measured_change"] for r in ok])
    visible = 0.10  # app.api.infection.VISIBLE_CHANGE
    tp = int(((t >= visible) & (m >= visible)).sum())
    fn = int(((t >= visible) & (m < visible)).sum())
    tn = int(((t < visible) & (m < visible)).sum())
    fp = int(((t < visible) & (m >= visible)).sum())
    by_amount = {}
    for amount in AMOUNTS:
        sel = [r for r in ok if r["amount"] == amount]
        by_amount[str(amount)] = {
            "n": len(sel),
            "true_mean": float(np.mean([r["true_change"] for r in sel])),
            "measured_mean": float(np.mean([r["measured_change"] for r in sel])),
            "mae": float(np.mean([abs(r["measured_change"] - r["true_change"]) for r in sel])),
        }
    nok = [r for r in nuisance_rows if r["measured_change"] is not None]
    summary = {
        "backgrounds_used": len({r["background"] for r in rows}),
        "backgrounds_skipped": skipped,
        "follow_ups": len(rows), "follow_ups_refused": len(rows) - len(ok),
        "pearson_r": float(np.corrcoef(t, m)[0, 1]),
        "mae_points": float(np.mean(np.abs(m - t)) * 100),
        "bias_points": float(np.mean(m - t) * 100),
        "visible_change_agreement": {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
                                     "sensitivity": tp / max(tp + fn, 1),
                                     "specificity": tn / max(tn + fp, 1)},
        "by_amount": by_amount,
        "nuisance": {
            "n": len(nuisance_rows), "refused": len(nuisance_rows) - len(nok),
            "false_visible_change": sum(r["measured_change"] >= visible for r in nok),
            "max_abs_change_points": float(max(abs(r["measured_change"]) for r in nok) * 100) if nok else None,
            "by_type": {
                name: {
                    "mean_abs_change_points": float(np.mean([abs(r["measured_change"]) for r in nok
                                                             if r["nuisance"] == name]) * 100),
                    "refused": sum(1 for r in nuisance_rows if r["nuisance"] == name and r["measured_change"] is None),
                } for name in sorted({r["nuisance"] for r in nuisance_rows})
                if any(r["nuisance"] == name for r in nok)
            },
        },
    }
    (DOCS / "site_monitoring_validation.json").write_text(json.dumps(
        {"summary": summary, "rows": rows, "nuisance_rows": nuisance_rows}, indent=2))

    # Scatter: measured vs true change.
    size = 420
    plot = np.full((size, size, 3), 255, np.uint8)
    cv2.line(plot, (40, size - 40), (size - 20, 20), (180, 180, 180), 1)
    for x in range(0, 101, 20):
        px = 40 + int(x / 100 * (size - 60))
        cv2.putText(plot, f"{x}", (px - 8, size - 22), 0, 0.35, (0, 0, 0), 1)
        cv2.putText(plot, f"{x}", (8, size - 40 - int(x / 100 * (size - 60)) + 4), 0, 0.35, (0, 0, 0), 1)
    for tv, mv in zip(t, m):
        cv2.circle(plot, (40 + int(tv * (size - 60)), size - 40 - int(np.clip(mv, 0, 1) * (size - 60))), 3, (200, 80, 30), -1)
    cv2.putText(plot, "true change (pts)", (150, size - 6), 0, 0.4, (0, 0, 0), 1)
    cv2.putText(plot, "measured", (44, 16), 0, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(DOCS / "figures/site_validation_scatter.png"), plot)

    # One row per background: baseline | change at +9 | change at +18.
    import base64 as b64
    def tile(data, label):
        img = cv2.resize(cv2.imdecode(np.frombuffer(b64.b64decode(data), np.uint8), 1), (300, 220))
        cv2.rectangle(img, (0, 0), (300, 24), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 17), 0, 0.45, (255, 255, 255), 1)
        return img
    rows_img = []
    for name in dict.fromkeys(e[0] for e in examples):
        ex = [e for e in examples if e[0] == name]
        row = [tile(ex[0][4], f"baseline: {name[:24]}")]
        for _, amount, truth, measured, _, change_b64 in ex:
            row.append(tile(change_b64, f"+{int(amount)} a*: true {truth:.0%} measured {measured:.0%}"))
        rows_img.append(np.hstack(row))
    if rows_img:
        cv2.imwrite(str(DOCS / "figures/site_validation_examples.png"), np.vstack(rows_img))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
