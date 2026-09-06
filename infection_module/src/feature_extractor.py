"""Colour and texture features for IV-site monitoring.

The central design constraint: **skin tone varies enormously between patients,
and lighting varies between visits.** An absolute redness measurement is
therefore close to meaningless across patients. Every colour feature here is
either a distribution statistic within one image, or a difference against a
patch of that same patient's healthy skin from that same photo. That makes the
features comparable across skin tones and lighting in a way raw RGB is not.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

log = logging.getLogger(__name__)

LBP_POINTS = 8
LBP_RADIUS = 1
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPERTIES = ["contrast", "homogeneity", "energy", "correlation"]


def get_baseline_patch(
    image: np.ndarray, roi_bbox: list[float], patch_frac: float = 0.25
) -> np.ndarray:
    """Sample healthy-looking skin just outside the ROI.

    This is the per-image reference that normalises for skin tone and lighting.
    Sampled from the band immediately outside the ROI so it is the same limb,
    the same illumination, and the same camera exposure.

    Tries each side in turn and takes the first that fits inside the frame, so a
    site photographed near the frame edge still yields a reference. Raises if no
    side fits - a fabricated reference would silently corrupt every colour
    distance downstream.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in roi_bbox)

    box_w, box_h = x2 - x1, y2 - y1
    patch_w = max(4, int(box_w * patch_frac))
    patch_h = max(4, int(box_h * patch_frac))
    gap = max(2, int(0.1 * max(box_w, box_h)))

    candidates = [
        (x1, y1 - gap - patch_h, x1 + patch_w, y1 - gap),        # above
        (x1, y2 + gap, x1 + patch_w, y2 + gap + patch_h),        # below
        (x1 - gap - patch_w, y1, x1 - gap, y1 + patch_h),        # left
        (x2 + gap, y1, x2 + gap + patch_w, y1 + patch_h),        # right
    ]

    for px1, py1, px2, py2 in candidates:
        if px1 >= 0 and py1 >= 0 and px2 <= width and py2 <= height and px2 > px1 and py2 > py1:
            return image[py1:py2, px1:px2]

    raise ValueError(
        "no healthy-skin patch fits outside the ROI - the site fills the frame. "
        "Retake the photo with more surrounding skin visible."
    )


def color_features(roi: np.ndarray, baseline_patch: np.ndarray) -> dict[str, float]:
    """HSV and LAB statistics, plus the erythema distance from healthy skin.

    `color_lab_distance` is the headline metric: Euclidean distance in LAB
    between mean ROI colour and mean healthy-skin colour. LAB is used because
    it is approximately perceptually uniform, so a fixed distance means roughly
    the same perceived colour change regardless of where in the space it sits -
    which RGB emphatically does not give you. This mirrors the approach in
    published cellulitis-tracking work.

    `color_a_delta` is broken out separately: a* is the green-to-red axis, so it
    isolates the specifically *red* component of any change. A bruise darkens
    without reddening and moves L more than a*; that distinction matters.
    """
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    patch_lab = cv2.cvtColor(baseline_patch, cv2.COLOR_BGR2LAB)

    features = {
        "color_hue_mean": float(roi_hsv[:, :, 0].mean()),
        "color_hue_std": float(roi_hsv[:, :, 0].std()),
        "color_sat_mean": float(roi_hsv[:, :, 1].mean()),
        "color_sat_std": float(roi_hsv[:, :, 1].std()),
        "color_a_mean": float(roi_lab[:, :, 1].mean()),
        "color_a_std": float(roi_lab[:, :, 1].std()),
        "color_b_mean": float(roi_lab[:, :, 2].mean()),
        "color_b_std": float(roi_lab[:, :, 2].std()),
    }

    roi_mean_lab = roi_lab.reshape(-1, 3).mean(axis=0)
    patch_mean_lab = patch_lab.reshape(-1, 3).mean(axis=0)

    features["color_lab_distance"] = float(np.linalg.norm(roi_mean_lab - patch_mean_lab))
    features["color_a_delta"] = float(roi_mean_lab[1] - patch_mean_lab[1])
    features["color_b_delta"] = float(roi_mean_lab[2] - patch_mean_lab[2])
    features["color_l_delta"] = float(roi_mean_lab[0] - patch_mean_lab[0])
    return features


def texture_features(roi_gray: np.ndarray) -> dict[str, float]:
    """LBP histogram and GLCM properties.

    Texture is what separates inflammation from a shadow or a skin-tone
    difference: inflamed tissue is taut and shiny with reduced fine texture,
    while normal skin retains its wrinkle and pore structure. A purely
    colour-based model would be far easier to fool with bad lighting.

    LBP is quantised to 8 points / radius 1 with the uniform variant, giving 10
    rotation-invariant bins. GLCM is averaged over four orientations for the
    same reason - a forearm can be photographed at any rotation.
    """
    if roi_gray.ndim == 3:
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

    features: dict[str, float] = {}

    lbp = local_binary_pattern(roi_gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2  # uniform method yields P+2 distinct codes
    histogram, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    for i, value in enumerate(histogram):
        features[f"texture_lbp_{i}"] = float(value)

    # GLCM on 32 grey levels rather than 256: with a small ROI, a 256x256
    # co-occurrence matrix is mostly empty and its statistics become noise.
    quantised = (roi_gray / 256.0 * 32).astype(np.uint8)
    glcm = graycomatrix(
        quantised, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        levels=32, symmetric=True, normed=True,
    )
    for prop in GLCM_PROPERTIES:
        values = graycoprops(glcm, prop)
        for d_idx, distance in enumerate(GLCM_DISTANCES):
            # Averaged over angles for rotation invariance.
            features[f"texture_glcm_{prop}_d{distance}"] = float(values[d_idx].mean())

    return features


def extract_all_features(
    image: np.ndarray, roi_bbox: list[float], baseline_patch: np.ndarray | None = None
) -> dict[str, float]:
    """Combined colour + texture feature dict with prefixed keys.

    `baseline_patch` is sampled automatically from `image` when not supplied.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in roi_bbox)
    roi = image[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        raise ValueError(f"ROI {roi_bbox} is empty on a {image.shape[:2]} image")

    if baseline_patch is None:
        baseline_patch = get_baseline_patch(image, roi_bbox)

    features = color_features(roi, baseline_patch)
    features.update(texture_features(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
    return features


def feature_names() -> list[str]:
    """Stable, ordered feature names.

    The risk model indexes a plain vector, so ordering must not drift between
    training and inference. Derived from a dummy extraction so it cannot fall
    out of sync with the functions above.
    """
    dummy = np.full((40, 40, 3), 150, dtype=np.uint8)
    return sorted(extract_all_features(dummy, [10, 10, 30, 30], dummy[:8, :8]).keys())
