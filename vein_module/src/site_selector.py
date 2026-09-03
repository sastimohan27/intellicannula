"""Pick an insertion site from a predicted vein mask.

This is the module whose output a clinician would act on, so it is written to
fail loudly and to report its own uncertainty rather than always returning an
answer. `select_insertion_site` returns confidence 0.0 and a `reason` when it
cannot find a defensible site; callers must check it.

Nothing here is validated against clinical outcomes. It encodes conventional
venipuncture heuristics - wider is better, straighter is better, away from the
frame edge is better - not evidence from a trial.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

log = logging.getLogger(__name__)

# Angle bounds in degrees. Shallow for superficial veins, steeper for deeper
# ones. Standard peripheral IV teaching is 10-30 degrees.
MIN_ANGLE_DEG = 15.0
MAX_ANGLE_DEG = 30.0

# A vein narrower than this is not a credible cannulation target at 512x512.
MIN_VEIN_WIDTH_PX = 3.0
MIN_SEGMENT_LENGTH_PX = 15.0

# Sliding-window linearity threshold from the plan.
LINEARITY_R2 = 0.95
WINDOW_SIZE = 9


def _order_skeleton_branch(points: np.ndarray) -> np.ndarray:
    """Order scattered skeleton pixels into a path by nearest-neighbour walk.

    `np.argwhere` returns raster order, which zig-zags across a diagonal vein
    and would make any sliding window meaningless. Starting from the point
    furthest from the centroid approximates starting at an endpoint.
    """
    if len(points) <= 2:
        return points

    remaining = points.tolist()
    centroid = points.mean(axis=0)
    start_idx = int(np.argmax(((points - centroid) ** 2).sum(axis=1)))
    ordered = [remaining.pop(start_idx)]

    while remaining:
        current = np.array(ordered[-1])
        distances = ((np.array(remaining) - current) ** 2).sum(axis=1)
        ordered.append(remaining.pop(int(np.argmin(distances))))

    return np.array(ordered)


def _r_squared(window: np.ndarray) -> float:
    """R^2 of a least-squares line fit to a run of skeleton points.

    Fits along whichever axis spans further, so a vertical vein is not scored
    as unfittable - a y-on-x fit of a vertical line has infinite slope and
    would spuriously reject a perfectly straight segment.
    """
    y, x = window[:, 0].astype(float), window[:, 1].astype(float)
    if np.ptp(x) >= np.ptp(y):
        independent, dependent = x, y
    else:
        independent, dependent = y, x

    variance = ((dependent - dependent.mean()) ** 2).sum()
    if variance == 0:
        return 1.0  # perfectly straight along the independent axis

    slope, intercept = np.polyfit(independent, dependent, 1)
    residual = ((dependent - (slope * independent + intercept)) ** 2).sum()
    return float(1.0 - residual / variance)


def _straight_segments(
    branch: np.ndarray, window_size: int = WINDOW_SIZE, r2_threshold: float = LINEARITY_R2
) -> list[np.ndarray]:
    """Longest maximal runs of the branch that stay linear.

    Overlapping windows are merged into maximal runs, so a long straight vein
    yields one long candidate rather than many short ones - otherwise the
    length term in the score would be capped at the window size.
    """
    if len(branch) < window_size:
        return [branch] if len(branch) >= 2 and _r_squared(branch) >= r2_threshold else []

    linear_at = [
        _r_squared(branch[i : i + window_size]) >= r2_threshold
        for i in range(len(branch) - window_size + 1)
    ]

    segments, start = [], None
    for i, is_linear in enumerate([*linear_at, False]):
        if is_linear and start is None:
            start = i
        elif not is_linear and start is not None:
            segments.append(branch[start : i + window_size - 1])
            start = None
    return segments


def select_insertion_site(
    mask: np.ndarray,
    min_width_px: float = MIN_VEIN_WIDTH_PX,
    min_length_px: float = MIN_SEGMENT_LENGTH_PX,
) -> dict:
    """Choose the best insertion point on a binary vein mask.

    Score per candidate straight segment:

        0.5 * normalized_width
      + 0.3 * normalized_length
      + 0.2 * (1 - proximity_to_image_edge)

    Returns
    -------
    dict with keys:
        x, y                 insertion point, or None when no site was found
        vein_width_px        local vein diameter at that point
        segment_length_px    length of the straight run it sits on
        confidence           0-1, margin of the winner over the runner-up
        reason               why confidence is low or no site was returned

    Confidence is a *separation* measure, not a probability of clinical
    success. A single candidate scores 0.5, not 1.0: one option is not
    evidence that the option is good.
    """
    binary = (mask > 0).astype(np.uint8)
    empty = {
        "x": None,
        "y": None,
        "vein_width_px": 0.0,
        "segment_length_px": 0.0,
        "confidence": 0.0,
    }

    if binary.sum() == 0:
        return {**empty, "reason": "empty mask - no veins segmented"}

    height, width = binary.shape
    # Distance transform gives the radius of the largest circle fitting inside
    # the vein at each pixel; on the skeleton that radius is the local half-width.
    distance = ndimage.distance_transform_edt(binary)
    skeleton = skeletonize(binary.astype(bool))

    if skeleton.sum() == 0:
        return {**empty, "reason": "mask too small to skeletonize"}

    labelled, n_branches = ndimage.label(skeleton, structure=np.ones((3, 3)))
    if n_branches == 0:
        return {**empty, "reason": "no connected vein branches"}

    centre = np.array([height / 2.0, width / 2.0])
    max_centre_distance = np.hypot(*centre)

    candidates = []
    for label in range(1, n_branches + 1):
        points = np.argwhere(labelled == label)
        if len(points) < 2:
            continue
        branch = _order_skeleton_branch(points)

        for segment in _straight_segments(branch):
            if len(segment) < 2:
                continue
            length = float(np.hypot(*(segment[-1] - segment[0])))
            if length < min_length_px:
                continue

            widths = distance[segment[:, 0], segment[:, 1]] * 2.0
            mean_width = float(widths.mean())
            if mean_width < min_width_px:
                continue

            midpoint = segment[len(segment) // 2]
            edge_proximity = float(
                np.hypot(*(midpoint - centre)) / max_centre_distance
            )
            candidates.append(
                {
                    "y": int(midpoint[0]),
                    "x": int(midpoint[1]),
                    "width": mean_width,
                    "length": length,
                    "edge_proximity": min(edge_proximity, 1.0),
                }
            )

    if not candidates:
        return {
            **empty,
            "reason": (
                f"no segment met the minimum bar "
                f"(width >= {min_width_px}px, straight length >= {min_length_px}px)"
            ),
        }

    max_width = max(c["width"] for c in candidates)
    max_length = max(c["length"] for c in candidates)
    for c in candidates:
        c["score"] = (
            0.5 * (c["width"] / max_width)
            + 0.3 * (c["length"] / max_length)
            + 0.2 * (1.0 - c["edge_proximity"])
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    if len(candidates) == 1:
        # One option is not evidence of a good option.
        confidence, reason = 0.5, "only one candidate segment - no alternative to compare"
    else:
        margin = (best["score"] - candidates[1]["score"]) / best["score"]
        confidence = float(np.clip(margin * 2.0, 0.0, 1.0))
        reason = (
            "clear winner"
            if confidence > 0.5
            else f"{len(candidates)} candidates scored similarly - low separation"
        )

    return {
        "x": best["x"],
        "y": best["y"],
        "vein_width_px": round(best["width"], 2),
        "segment_length_px": round(best["length"], 2),
        "confidence": round(confidence, 3),
        "reason": reason,
        "n_candidates": len(candidates),
    }


def estimate_needle_angle(
    vein_width_px: float,
    contrast_intensity: float,
    width_range: tuple[float, float] = (3.0, 20.0),
) -> float:
    """Map apparent width and contrast to a suggested insertion angle.

    HEURISTIC PROXY FOR DEPTH - NOT A MEASUREMENT.
    ----------------------------------------------
    A single 2D image carries no depth information. This assumes that a vein
    appearing wider and higher-contrast is more superficial, because deeper
    vessels are blurred and attenuated by the tissue above them. That
    assumption fails in obvious ways: a large deep vein can look wider than a
    small superficial one, and skin tone, hydration, and lighting all shift
    apparent contrast independently of depth.

    Consequence of being wrong: too shallow risks a failed stick, too steep
    risks passing through the posterior wall. Must be validated against
    ultrasound-measured depth before any clinical use, and the UI must present
    this as a suggestion a clinician overrides, never an instruction.

    Returns degrees, clamped to [15, 30].
    """
    low, high = width_range
    normalized_width = np.clip((vein_width_px - low) / (high - low), 0.0, 1.0)
    normalized_contrast = np.clip(contrast_intensity, 0.0, 1.0)

    # Equal weighting - there is no evidence to justify preferring one cue.
    superficiality = 0.5 * normalized_width + 0.5 * normalized_contrast

    angle = MAX_ANGLE_DEG - superficiality * (MAX_ANGLE_DEG - MIN_ANGLE_DEG)
    return float(round(np.clip(angle, MIN_ANGLE_DEG, MAX_ANGLE_DEG), 1))
