"""Bridge between the API and the two ML modules.

Both pipelines are loaded lazily and cached. Loading a 31 MB checkpoint on
every request would dominate endpoint latency, and importing torch at module
scope would make the API slow to start even when no ML call is made.

Everything here degrades to a clear error rather than a wrong answer: a missing
checkpoint raises with instructions, and the vein pipeline returning "no site"
is passed through as a real result rather than being papered over.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path

# --- OpenMP: must be set before torch or lightgbm are imported ---
#
# PyTorch and LightGBM each bundle their own OpenMP runtime. Loading both into
# one process on macOS segfaults the interpreter - verified here: a vein scan
# (torch) followed by a site check (lightgbm) crashed uvicorn with SIGSEGV
# immediately after the first request succeeded.
#
# Pinning both to a single OpenMP thread avoids the duplicate thread pools that
# trigger it. KMP_DUPLICATE_LIB_OK permits the second runtime to load at all.
# The cost is no intra-op parallelism, which is acceptable: inference here is
# one image at a time, and correctness beats throughput in a clinical path.
#
# The durable fix is to run the two modules as separate services - which is what
# the module Dockerfiles are for, and what a real deployment should do. This
# keeps the single-process dev stack working in the meantime.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from app.core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# The ML modules live outside the backend package.
if str(settings.repo_root) not in sys.path:
    sys.path.insert(0, str(settings.repo_root))

_vein_model = None
_tracker = None


class ModelNotAvailable(RuntimeError):
    """Raised when a checkpoint is missing, with how to produce it."""


def _decode_image(data: bytes) -> np.ndarray:
    import cv2

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image - is it a valid PNG or JPEG?")
    return image


def _encode_png(image: np.ndarray) -> str:
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode overlay PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def get_vein_model():
    global _vein_model
    if _vein_model is None:
        import torch

        torch.set_num_threads(1)

        from vein_module.src.model import AttentionUNet

        path = settings.repo_root / "vein_module/models/attention_unet_best.pt"
        if not path.exists():
            raise ModelNotAvailable(
                f"no vein checkpoint at {path}. Train one with "
                "`python vein_module/src/train.py`."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        model = AttentionUNet(base_channels=config.get("base_channels", 32))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _vein_model = (model, config.get("image_size", 512))
        log.info("loaded vein model from %s (epoch %s)", path, checkpoint.get("epoch"))
    return _vein_model


def run_vein_scan(image_bytes: bytes) -> dict:
    """Preprocess, segment, choose an insertion site, estimate an angle.

    Returns the recommendation plus a base64 overlay for display. A refusal
    (`insertion_x` None) is a legitimate result and is returned as such.
    """
    import cv2
    import torch

    from vein_module.src.augmentations import get_eval_transform
    from vein_module.src.site_selector import estimate_needle_angle, select_insertion_site

    model, image_size = get_vein_model()
    original = _decode_image(image_bytes)

    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    transformed = get_eval_transform(image_size)(image=rgb)["image"]
    tensor = torch.from_numpy(
        transformed.astype(np.float32).transpose(2, 0, 1) / 255.0
    ).unsqueeze(0)

    with torch.inference_mode():
        probs = torch.sigmoid(model(tensor)).numpy()[0, 0]
    mask = (probs > 0.5).astype(np.uint8)

    site = select_insertion_site(mask)

    angle = None
    if site["x"] is not None:
        # Contrast proxy: how strongly the vein stands out from its frame.
        gray = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
        vein_pixels = gray[mask.astype(bool)]
        contrast = (
            float(abs(vein_pixels.mean() - gray.mean()) / 128.0)
            if vein_pixels.size else 0.0
        )
        angle = estimate_needle_angle(site["vein_width_px"], min(contrast, 1.0))

    resized = cv2.resize(original, (image_size, image_size), interpolation=cv2.INTER_AREA)
    overlay = _draw_overlay(resized, mask, site)

    return {
        "insertion_x": site["x"],
        "insertion_y": site["y"],
        "angle_deg": angle,
        "vein_width_px": site["vein_width_px"] or None,
        "confidence": site["confidence"],
        "reason": site.get("reason"),
        "mask_png": cv2.imencode(".png", mask * 255)[1].tobytes(),
        "overlay_png_base64": _encode_png(overlay),
        "model_version": settings.vein_model_version,
    }


def _draw_overlay(image: np.ndarray, mask: np.ndarray, site: dict) -> np.ndarray:
    import cv2

    canvas = image.copy()
    tint = np.zeros_like(canvas)
    tint[mask.astype(bool)] = (0, 0, 255)
    canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0)
    if site["x"] is not None:
        cv2.drawMarker(canvas, (site["x"], site["y"]), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(canvas, (site["x"], site["y"]), 18, (0, 255, 0), 2)
    return canvas


def get_tracker():
    """Process-wide tracker holding each patient's baseline and series.

    In-memory by design: the DB is the durable record, this is the working set
    the alert rules read so an upload does not wait on a round trip. It is
    rebuilt from the database on restart via `seed_tracker`.
    """
    global _tracker
    if _tracker is None:
        from infection_module.src.temporal_tracker import PatientSiteTracker

        _tracker = PatientSiteTracker()
    return _tracker


def run_site_check(image_bytes: bytes, patient_id: int, baseline_bytes: bytes | None) -> dict:
    """Detect the hub, register against baseline, extract features, score risk.

    ROI is located twice on purpose. Registration warps the new photo into the
    *baseline's* coordinate frame, so it comes back with the baseline's
    dimensions - which may differ from the photo just uploaded. An ROI measured
    on the original image is meaningless afterwards, and using it raised
    "ROI [140, 91.5, 420, 274.5] is empty on a (80, 94) image" the first time
    the two differed in size.

    So: locate once on the original to anchor the translation fallback, then
    locate again on the aligned image, which is the frame the features are
    actually measured in.
    """
    from infection_module.src.feature_extractor import extract_all_features
    from infection_module.src.registration import register_images
    from infection_module.src.risk_model import predict_risk

    image = _decode_image(image_bytes)
    anchor_bbox = _locate_roi(image)

    degraded = False
    aligned = image
    if baseline_bytes is not None:
        baseline = _decode_image(baseline_bytes)
        aligned, info = register_images(
            baseline, image, roi_bbox=anchor_bbox, return_info=True
        )
        degraded = info["degraded"]

    roi_bbox = _locate_roi(aligned) if aligned.shape[:2] != image.shape[:2] else anchor_bbox

    try:
        features = extract_all_features(aligned, roi_bbox)
    except ValueError as exc:
        raise ValueError(
            f"{exc} Capture the site with more surrounding skin visible, and "
            "keep framing consistent with the baseline photo."
        ) from exc

    model_path = settings.repo_root / "infection_module/models/risk_lgbm.txt"
    if not model_path.exists():
        raise ModelNotAvailable(
            f"no risk model at {model_path}. Train one with "
            "`python infection_module/src/risk_model.py --train`."
        )
    risk = predict_risk(features, model_path)

    return {
        "risk_score": risk,
        "roi_bbox": roi_bbox,
        "color_distance": features["color_lab_distance"],
        "registration_degraded": degraded,
        "model_version": settings.infection_model_version,
    }


def _locate_roi(image: np.ndarray) -> list[float]:
    """Find the monitoring ROI, preferring the cannula detector.

    Falls back to the central region when no detector weights exist - Phase 4
    needs hand-labelled data that is not public. The fallback is honest rather
    than silent: it is logged, and it assumes the photographer centred the site,
    which is what the capture instructions ask for.
    """
    from infection_module.src.cannula_detector.inference import crop_roi  # noqa: F401

    weights = settings.repo_root / "infection_module/models/cannula_yolo11.pt"
    if weights.exists():
        import tempfile

        import cv2

        from infection_module.src.cannula_detector.inference import detect_cannula

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            cv2.imwrite(handle.name, image)
            detection = detect_cannula(handle.name, weights=weights)
        if detection is not None:
            x1, y1, x2, y2 = detection["bbox"]
            mx, my = (x2 - x1) * 0.4, (y2 - y1) * 0.4
            height, width = image.shape[:2]
            return [
                max(0.0, x1 - mx), max(0.0, y1 - my),
                min(float(width), x2 + mx), min(float(height), y2 + my),
            ]
        log.info("cannula detector found no hub; using centre ROI")
    else:
        log.debug("no cannula detector weights; using centre ROI")

    height, width = image.shape[:2]
    return [width * 0.25, height * 0.25, width * 0.75, height * 0.75]
