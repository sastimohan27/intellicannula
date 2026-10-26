"""Turn the network's probability map into the binary vein mask.

Shared by the API (`backend/app/core/ml.py`) and `evaluate.py`, so the metrics
in the evaluation report are measured on exactly the mask the API serves.
"""

from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Drop 8-connected blobs smaller than `min_area` pixels; returns uint8 0/1.

    Removes isolated speckle only. Any component at or above `min_area` is kept
    whole, so a thin but continuous vein survives regardless of its width.
    """
    binary = np.asarray(binary).astype(np.uint8)
    if min_area <= 0:
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def binarize(probs: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    """Threshold a probability map, then remove speckle."""
    return remove_small_components(probs > threshold, min_area)


def hysteresis(probs: np.ndarray, high: float, low: float, min_area: int) -> np.ndarray:
    """Keep low-threshold pixels only where they connect to a high-threshold seed.

    Aimed at thin veins: their faint continuations fall below a single strict
    threshold, while isolated low-probability speckle has no confident seed.
    Experimental - see vein_module/src/postprocess_experiments.py for whether
    it is adopted.
    """
    weak = (probs > low).astype(np.uint8)
    strong = probs > high
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return remove_small_components(keep[labels], min_area)
