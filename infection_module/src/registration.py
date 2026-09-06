"""Align a follow-up photo to the patient's baseline.

Without registration, a "redness increased 12%" measurement is confounded by
the camera simply being held closer or at a different angle. This module makes
successive visits comparable, and - importantly - reports when it could not.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_GOOD_MATCHES = 10
RATIO_TEST = 0.75


def register_images(
    baseline: np.ndarray,
    current: np.ndarray,
    roi_bbox: list[float] | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Warp `current` into `baseline`'s frame.

    ORB keypoints, brute-force Hamming matching with Lowe's ratio test, then a
    RANSAC homography.

    Fallback. With fewer than 10 good matches - common on a smooth, featureless
    forearm - a homography fitted anyway would be wildly wrong, and a wrong warp
    is worse than no warp because it silently moves the ROI onto different
    tissue. So the fallback is a *translation only*, derived from the cannula
    bbox centre when one is supplied, and the returned info records that this
    happened. Callers should surface `method` to the clinician: a
    translation-only alignment means cross-visit comparisons are weaker.

    Returns the warped image, or `(warped, info)` when `return_info=True`.
    """
    info = {"method": "homography", "n_matches": 0, "degraded": False}

    baseline_gray = _as_gray(baseline)
    current_gray = _as_gray(current)

    orb = cv2.ORB_create(nfeatures=2000)
    kp_base, desc_base = orb.detectAndCompute(baseline_gray, None)
    kp_curr, desc_curr = orb.detectAndCompute(current_gray, None)

    good: list = []
    if desc_base is not None and desc_curr is not None and len(kp_base) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # knnMatch needs k=2 neighbours for the ratio test; guard tiny sets.
        if len(desc_base) >= 2:
            for pair in matcher.knnMatch(desc_curr, desc_base, k=2):
                if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance:
                    good.append(pair[0])

    info["n_matches"] = len(good)

    if len(good) >= MIN_GOOD_MATCHES:
        src = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_base[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if homography is not None:
            info["inliers"] = int(mask.sum()) if mask is not None else 0
            warped = cv2.warpPerspective(
                current, homography, (baseline.shape[1], baseline.shape[0])
            )
            return (warped, info) if return_info else warped
        log.warning("RANSAC found no consistent homography despite %d matches", len(good))

    log.warning(
        "only %d good ORB matches (need %d) - falling back to translation-only "
        "alignment; cross-visit comparison is degraded",
        len(good),
        MIN_GOOD_MATCHES,
    )
    info.update({"method": "translation_fallback", "degraded": True})
    warped = _translate_to_baseline(baseline, current, roi_bbox)
    return (warped, info) if return_info else warped


def _as_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _translate_to_baseline(
    baseline: np.ndarray, current: np.ndarray, roi_bbox: list[float] | None
) -> np.ndarray:
    """Translation-only alignment.

    Centres `current` on the baseline frame, using the cannula bbox centre as
    the anchor when available and the image centre otherwise. Deliberately does
    not rotate or scale - with too few matches there is no reliable estimate of
    either, and inventing one would be worse than admitting the limitation.
    """
    height, width = baseline.shape[:2]

    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        anchor = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    else:
        anchor = np.array([current.shape[1] / 2.0, current.shape[0] / 2.0])

    target = np.array([width / 2.0, height / 2.0])
    shift = target - anchor

    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(current, matrix, (width, height))
