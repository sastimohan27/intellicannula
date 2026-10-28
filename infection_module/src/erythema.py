"""Peri-cannula redness (erythema) measurement.

Replaces the LightGBM "risk" in the API. That model was trained to separate
close-up chronic wounds from normal skin (AZH proxy labels), so on real IV
photos - tape, plastic hub and dressing inside the box - it scored ~0.999 even
on a clean site. It answered "is this a wound?", not "is this site reddening?".

This module measures the clinically visible sign directly. Phlebitis presents
as redness spreading out from the insertion point, so:

1. **Ring.** Skin in an annulus around the detected hub, from `RING_INNER` to
   `RING_OUTER` hub half-sizes from its centre. The hub itself is excluded.
2. **Skin only.** A YCrCb skin mask (Chai & Ngan ranges) drops dressing, tape,
   the coloured plastic hub and background, so they cannot read as "red".
3. **Self reference.** Median a* of the farthest visible skin in the same photo.
   Differences within one photo cancel exposure, white balance and skin tone,
   which absolute colour values do not.
4. **Per-pixel Δa*.** a* is CIELAB's green-red axis; a bruise or shadow moves
   L* far more than a*, which is why a* rather than overall colour is used.
5. **Scores.** `reddened_fraction` = share of ring skin with Δa* above
   `RED_DELTA_A`; `spread_hub_widths` = how far the reddened skin reaches.

Units: OpenCV 8-bit LAB stores a* + 128, so Δa* here is in CIELAB a* units.
`RED_DELTA_A = 6` sits above the ~2-3 unit just-noticeable difference reported
for skin colorimetry, so ordinary skin variation does not count as redness.

NOT CLINICALLY VALIDATED. The thresholds and the redness level are documented
heuristics. They show that the measurement responds correctly to redness
(see tests); whether they predict phlebitis needs nurse-scored IV-site photos.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

RING_INNER = 0.75       # x hub half-size from the hub centre: just outside the hub
RING_OUTER = 5.0        # outer edge of the measured ring (~2.5 hub widths)
REF_QUANTILE = 0.7      # reference = the farthest 30% of visible skin from the hub.
                        # It must sit well clear of the site: a fixed band at 2.5-3.5
                        # half-sizes let spreading redness redden the reference too,
                        # and the score FELL as redness was added (caught by the
                        # functional test). A fixed far distance (6 half-sizes) fell
                        # outside the frame on close-up phone photos, so the farthest
                        # skin actually in the photo is used instead.
MIN_REF_DISTANCE = 2.5  # refuse if even the farthest skin is this close to the hub
RED_DELTA_A = 6.0       # Δa* above which a skin pixel counts as reddened
MIN_SKIN_FRACTION = 0.15  # of the ring; below this there is too little skin to judge
MIN_REF_PIXELS = 200
MAX_SIDE = 1024         # images are downscaled to this for speed and stable scale

# Redness level: documented heuristic bands on reddened_fraction, not VIP.
# VIP also needs pain, swelling and a palpable cord, which a photo cannot show.
LEVELS = [
    (0.10, 0, "No visible redness"),
    (0.25, 1, "Mild redness"),
    (0.50, 2, "Marked redness"),
    (1.01, 3, "Extensive redness"),
]


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean skin mask in YCrCb, minus near-white, near-black and specular pixels."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135) & (y > 40) & (y < 240)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    skin &= hsv[..., 1] > 25            # tape, dressing and gauze are near-grey
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return skin.astype(bool)


def _radial_distance(shape: tuple[int, int], bbox: list[float]) -> np.ndarray:
    """Distance from the hub centre in units of hub half-size (Chebyshev-like box metric).

    A box metric matches the bbox shape, so the ring hugs an elongated hub
    instead of being a circle that is too tight on one axis.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max((x2 - x1) / 2, 1.0), max((y2 - y1) / 2, 1.0)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.maximum(np.abs(xx - cx) / hw, np.abs(yy - cy) / hh)


def _downscale(bgr: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, list[float]]:
    scale = MAX_SIDE / max(bgr.shape[:2])
    if scale >= 1.0:
        return bgr, list(bbox)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, [v * scale for v in bbox]


def measure(bgr: np.ndarray, hub_bbox: list[float]) -> dict:
    """Redness around one detected hub. Raises ValueError when it cannot judge."""
    bgr, hub_bbox = _downscale(bgr, hub_bbox)
    r = _radial_distance(bgr.shape[:2], hub_bbox)
    skin = skin_mask(bgr)

    ring = (r >= RING_INNER) & (r < RING_OUTER)
    ring_skin = ring & skin

    ring_px = int(ring.sum())
    skin_fraction = float(ring_skin.sum()) / ring_px if ring_px else 0.0
    if skin_fraction < MIN_SKIN_FRACTION:
        raise ValueError(
            f"Only {skin_fraction:.0%} of the area around the cannula is visible skin "
            f"(need {MIN_SKIN_FRACTION:.0%}). Retake with less dressing or tape "
            "covering the surrounding skin."
        )

    candidates = skin & (r >= RING_INNER)
    if candidates.sum() < MIN_REF_PIXELS:
        raise ValueError(
            "Not enough skin visible around the cannula to use as a reference. "
            "Step back so more of the limb is in frame."
        )
    cutoff = float(np.quantile(r[candidates], REF_QUANTILE))
    if cutoff < MIN_REF_DISTANCE:
        raise ValueError(
            "All visible skin is right next to the cannula, so there is no normal "
            "skin to compare against. Step back so more of the limb is in frame."
        )
    reference = candidates & (r >= cutoff)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    ref_a = float(np.median(a[reference]))
    ref_b = float(np.median(lab[..., 2][reference] - 128.0))
    delta_a = a - ref_a
    red = ring_skin & (delta_a > RED_DELTA_A)

    fraction = float(red.sum()) / float(ring_skin.sum())
    spread = float(np.percentile(r[red], 90)) if red.sum() >= 20 else 0.0
    level, label = next((lvl, lab) for upper, lvl, lab in LEVELS if fraction < upper)

    return {
        "reddened_fraction": fraction,
        "mean_delta_a": float(delta_a[ring_skin].mean()),
        "spread_hub_widths": spread / 2.0,   # r is in half-sizes
        "skin_fraction": skin_fraction,
        "reference_a": ref_a,
        "reference_b": ref_b,
        "reference_distance_hub_widths": cutoff / 2.0,
        "redness_level": level,
        "redness_label": label,
        "_bgr": bgr, "_ring": ring, "_red": red, "_hub": hub_bbox,
    }


def risk_score(fraction: float) -> float:
    """0-1 score for the trend chart and alert rules: reddened ring share, saturating at 50%.

    With the API's alert threshold of 0.6 this alerts when ~30% of the
    surrounding skin is reddened. Linear and documented, so anyone can check it.
    """
    return float(np.clip(fraction / 0.5, 0.0, 1.0))


def overlay_png_base64(result: dict) -> str:
    """Photo with the hub box, the measured ring, and reddened skin tinted."""
    canvas = result["_bgr"].copy()
    tint = canvas.copy()
    tint[result["_red"]] = (0, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.55, tint, 0.45, 0)

    ring_edge = result["_ring"].astype(np.uint8)
    contours, _ = cv2.findContours(ring_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)
    x1, y1, x2, y2 = (int(round(v)) for v in result["_hub"])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("failed to encode overlay PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def public(result: dict) -> dict:
    """Result without the private arrays used for drawing."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


# --- Baseline-anchored change (candidate method, see scripts/validate_site_monitoring.py) ---
#
# The in-photo reference above cancels lighting, but redness that spreads to the
# far skin raises the reference too and cancels itself: on real limb photos with
# injected redness it under-measured change by ~16 points. This method compares
# the follow-up with the BASELINE pixel by pixel instead, after (1) aligning it to
# the baseline (the homography from the scene check) and (2) mapping its colours
# onto the baseline's using neutral pixels - sheets, tape, dressing - which are
# not skin and should not change colour between visits.

NEUTRAL_MAX_SAT = 30
NEUTRAL_MIN_VALUE = 110
NEUTRAL_MIN_FRACTION = 0.005


def _neutral_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < NEUTRAL_MAX_SAT) & (hsv[..., 2] > NEUTRAL_MIN_VALUE) & (hsv[..., 2] < 250)


def change_vs_baseline(base_bgr: np.ndarray, hub_bbox: list[float], aligned_bgr: np.ndarray) -> dict:
    """Share of the baseline's ring skin whose a* rose by >= RED_DELTA_A since baseline.

    `aligned_bgr` must already be warped into the baseline's frame (same shape).
    """
    if aligned_bgr.shape != base_bgr.shape:
        raise ValueError("follow-up must be aligned to the baseline frame first")
    base, hub = _downscale(base_bgr, hub_bbox)
    follow = cv2.resize(aligned_bgr, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)

    valid = follow.sum(axis=2) > 0                       # warp leaves black borders
    # Neutral objects are located in the BASELINE only: a colour cast in the
    # follow-up can push a white sheet past the saturation limit, and the
    # photos are aligned, so the same positions hold the same objects.
    # Clipped follow-up pixels carry no colour information and are excluded.
    # They must also lie OUTSIDE the measured ring: a transparent dressing over
    # the site looks white at baseline but reddens with the skin beneath it, and
    # normalising on it cancelled real change (validation r fell 0.99 -> 0.94).
    r = _radial_distance(base.shape[:2], hub)
    neutral = _neutral_mask(base) & valid & (follow.max(axis=2) < 250) & (r >= RING_OUTER)
    normalised = neutral.mean() >= NEUTRAL_MIN_FRACTION
    if normalised:
        gains = base[neutral].astype(np.float64).mean(0) / np.maximum(follow[neutral].astype(np.float64).mean(0), 1)
        follow = np.clip(follow.astype(np.float64) * gains, 0, 255).astype(np.uint8)

    region = (r >= RING_INNER) & (r < RING_OUTER) & skin_mask(base) & valid
    if region.sum() < MIN_REF_PIXELS:
        raise ValueError("Too little skin around the cannula overlaps between the two photos.")

    a_base = cv2.GaussianBlur(cv2.cvtColor(base, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32), (5, 5), 0)
    a_follow = cv2.GaussianBlur(cv2.cvtColor(follow, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32), (5, 5), 0)
    delta = a_follow - a_base
    return {
        "change": float((delta[region] >= RED_DELTA_A).mean()),
        "colour_normalised": bool(normalised),
        "neutral_fraction": float(neutral.mean()),
        "_delta": delta, "_region": region, "_bgr": follow,
    }


def change_overlay_png_base64(change: dict) -> str:
    """Aligned follow-up with pixels reddened since baseline (Δa* >= threshold) marked."""
    canvas = change["_bgr"].copy()
    hit = change["_region"] & (change["_delta"] >= RED_DELTA_A)
    tint = canvas.copy()
    tint[hit] = (255, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.5, tint, 0.5, 0)
    contours, _ = cv2.findContours(change["_region"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 1)
    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("failed to encode change overlay PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
