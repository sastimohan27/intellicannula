"""Cannula hub detection and ROI cropping.

The detector's job is to anchor the infection-monitoring ROI to the same
anatomical spot across visits, so that a redness measurement on day 3 is
comparable to day 1. Everything downstream in Phase 5 depends on that anchor
being stable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "infection_module/models/cannula_yolo11.pt"
_model_cache: dict[str, object] = {}


def _load_model(weights: Path):
    """Load and cache the YOLO model.

    Cached because Phase 6 calls this per request; reloading weights on every
    upload would dominate the endpoint's latency.
    """
    key = str(weights)
    if key not in _model_cache:
        from ultralytics import YOLO  # imported lazily - pulls in torch

        if not weights.exists():
            raise FileNotFoundError(
                f"no cannula detector at {weights}. Train one with "
                "`python infection_module/src/cannula_detector/train.py` "
                "(see docs/labeling_guide.md for the dataset it needs)."
            )
        _model_cache[key] = YOLO(str(weights))
    return _model_cache[key]


def detect_cannula(
    image_path: str | Path | np.ndarray,
    conf_threshold: float = 0.4,
    weights: Path | None = None,
) -> dict | None:
    """Highest-confidence cannula hub in an image.

    Returns `{"bbox": [x1, y1, x2, y2], "confidence": float}`, or **None** when
    nothing clears the threshold. None is a meaningful answer: no hub visible
    means the photo is unusable for site monitoring, and the caller should ask
    for a retake rather than measure a region that may not contain the site.
    """
    model = _load_model(weights or DEFAULT_WEIGHTS)
    # A BGR ndarray (OpenCV convention, which ultralytics expects) is passed
    # through, so the API need not write a temporary file per request.
    source = image_path if isinstance(image_path, np.ndarray) else str(image_path)
    results = model.predict(source, conf=conf_threshold, verbose=False)

    best = None
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if best is None or confidence > best["confidence"]:
                best = {
                    "bbox": [float(v) for v in box.xyxy[0].tolist()],
                    "confidence": confidence,
                }

    if best is None:
        log.info("no cannula hub above conf=%.2f", conf_threshold)
    return best


def crop_roi(
    image: np.ndarray, bbox: list[float], margin_frac: float = 0.4
) -> np.ndarray:
    """Crop around a bbox with proportional margin, clipped to image bounds.

    The margin matters clinically: phlebitis spreads *along the vein, away from*
    the insertion point, so a tight crop on the hub would miss the streaking
    that indicates progression. The margin also supplies surrounding skin for
    the per-image healthy-skin reference in `feature_extractor`.

    Margin is proportional to bbox size rather than fixed, so the ROI covers
    the same anatomical area whether the photo was taken close up or at arm's
    length.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox

    margin_x = (x2 - x1) * margin_frac
    margin_y = (y2 - y1) * margin_frac

    # Clip to bounds. A hub photographed near the frame edge yields an
    # asymmetric crop rather than an out-of-range index.
    cx1 = max(0, int(round(x1 - margin_x)))
    cy1 = max(0, int(round(y1 - margin_y)))
    cx2 = min(width, int(round(x2 + margin_x)))
    cy2 = min(height, int(round(y2 + margin_y)))

    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(
            f"bbox {bbox} produces an empty crop on a {width}x{height} image"
        )
    return image[cy1:cy2, cx1:cx2]
