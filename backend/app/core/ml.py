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
import functools
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


class NoCannulaDetected(ValueError):
    """The photo has no detectable cannula hub, so there is no site to score.

    A ValueError so the endpoint maps it to 422: scoring an arbitrary region of
    a photo with no cannula in it produced confident risk scores for images
    that were not IV sites at all.
    """


class PhotoQualityError(ValueError):
    """The photo is too blurred, dark or overexposed to measure; ask for a retake."""


# Measured on the demo and test photos: every usable photo scored sharpness
# >= 57 (variance of the Laplacian at <=1024 px), Gaussian-blurred copies <= 3.3.
MIN_SHARPNESS = 20.0
MIN_BRIGHTNESS = 40.0
MAX_OVEREXPOSED = 0.25


def check_photo_quality(image: np.ndarray) -> dict:
    """Reject photos that cannot be measured reliably, before any model runs."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1024 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    overexposed = float((gray > 250).mean())

    if sharpness < MIN_SHARPNESS:
        raise PhotoQualityError("Photo is blurred. Hold the phone steady and tap to focus on the cannula.")
    if brightness < MIN_BRIGHTNESS:
        raise PhotoQualityError("Photo is too dark. Turn on the room light (keep the flash off).")
    if overexposed > MAX_OVEREXPOSED:
        raise PhotoQualityError("Photo is overexposed. Turn off the flash and avoid direct light on the dressing.")
    return {"sharpness": sharpness, "brightness": brightness, "overexposed": overexposed}


def _requires_ml_deps(fn):
    """Turn a missing heavy dependency (torch, cv2, ...) into ModelNotAvailable.

    Without this the ImportError escapes as an unhandled 500, which skips the
    CORS middleware - so the browser reports "cannot reach the API" instead of
    the real problem.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ModuleNotFoundError as exc:
            raise ModelNotAvailable(
                f"ML dependency '{exc.name}' is not installed in this Python "
                f"({sys.executable}). Run the API from infection_module/.venv - "
                "see backend/requirements.txt."
            ) from exc

    return wrapper


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


@_requires_ml_deps
def run_vein_scan(image_bytes: bytes) -> dict:
    """Preprocess, segment, choose an insertion site, estimate an angle.

    Returns the recommendation plus a base64 overlay for display. A refusal
    (`insertion_x` None) is a legitimate result and is returned as such.
    """
    import cv2
    import torch

    from vein_module.src.augmentations import get_eval_transform
    from vein_module.src.postprocess import binarize
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
    mask = binarize(probs, settings.vein_mask_threshold, settings.vein_min_component_px)

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


_baseline_cache: dict[str, dict | None] = {}


@_requires_ml_deps
def run_site_check(
    image_bytes: bytes, patient_id: int, baseline_bytes: bytes | None
) -> dict:
    """Detect the hub, then measure redness in the skin ring around it.

    Scoring is `infection_module.src.erythema` (see its docstring for why the
    LightGBM proxy model was retired from this path). Each photo is measured
    around its *own* detected hub against skin further out in the *same* photo,
    so no cross-visit image warping is needed - which removes the failure where
    warping onto an unrelated baseline lost the hub. The baseline photo is
    measured the same way, and the change is a difference of two measurements.
    """
    from infection_module.src import erythema

    image = _decode_image(image_bytes)
    check_photo_quality(image)
    roi_bbox, cannula_confidence, hub = _locate_roi(image)
    result = erythema.measure(image, hub)
    current = erythema.public(result)

    baseline = _baseline_measurement(baseline_bytes) if baseline_bytes else None
    change = (
        current["reddened_fraction"] - baseline["reddened_fraction"]
        if baseline is not None else None
    )

    return {
        "risk_score": erythema.risk_score(current["reddened_fraction"]),
        "roi_bbox": roi_bbox,
        "color_distance": current["mean_delta_a"],
        "registration_degraded": False,
        "cannula_confidence": cannula_confidence,
        "redness": current,
        "baseline_reddened_fraction": (
            baseline["reddened_fraction"] if baseline is not None else None
        ),
        "change_since_baseline": change,
        "overlay_png_base64": erythema.overlay_png_base64(result),
        "model_version": settings.infection_model_version,
    }


def _baseline_measurement(baseline_bytes: bytes) -> dict | None:
    """Redness of the patient's first photo; None when it is unusable.

    Cached by content hash: the baseline never changes, and re-running the
    detector subprocess on it for every check-in would double the latency.
    A baseline with no detectable hub (stored before the cannula gate
    existed) yields None, and the change is reported as unavailable rather
    than computed against an arbitrary region.
    """
    import hashlib

    from infection_module.src import erythema

    key = hashlib.sha256(baseline_bytes).hexdigest()
    if key not in _baseline_cache:
        try:
            image = _decode_image(baseline_bytes)
            _, _, hub = _locate_roi(image)
            _baseline_cache[key] = erythema.public(erythema.measure(image, hub))
        except ValueError as exc:
            log.warning("baseline photo unusable for comparison: %s", exc)
            _baseline_cache[key] = None
    return _baseline_cache[key]


def _locate_roi(image: np.ndarray) -> tuple[list[float], float, list[float]]:
    """Find the monitoring ROI around the detected cannula hub.

    Returns `(roi_bbox, detector_confidence, hub_bbox)`; the ROI is the hub box
    plus a 40% margin, for display and storage. There is deliberately no
    fallback region: an earlier version scored the image centre whenever the
    detector found nothing (or crashed), which gave a landscape photo an
    infection risk of 0.999. No hub means no site, so this raises instead.

    The detector runs in a subprocess because ultralytics' torch and LightGBM
    each bundle OpenMP, and loading both in one process segfaults on macOS.
    """
    import json
    import subprocess
    import tempfile

    import cv2

    weights = settings.repo_root / "infection_module/models/cannula_yolo11.pt"
    if not weights.exists():
        raise ModelNotAvailable(
            f"no cannula detector at {weights}. Train one with "
            "`python infection_module/src/cannula_detector/train.py`."
        )

    threshold = settings.cannula_conf_threshold
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "site.png"
        cv2.imwrite(str(image_path), image)
        py_cmd = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from infection_module.src.cannula_detector.inference import detect_cannula\n"
            "res = detect_cannula(sys.argv[1], conf_threshold=float(sys.argv[3]),"
            " weights=Path(sys.argv[2]))\n"
            "print(json.dumps(res))\n"
        )
        env = {**os.environ, "PYTHONPATH": str(settings.repo_root)}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", py_cmd, str(image_path), str(weights), str(threshold)],
                capture_output=True, timeout=60, env=env, check=True,
            )
        except subprocess.CalledProcessError as exc:
            tail = exc.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["?"]
            raise ModelNotAvailable(f"cannula detector failed: {tail[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ModelNotAvailable("cannula detector timed out") from exc

    lines = proc.stdout.decode().strip().splitlines()
    detection = json.loads(lines[-1]) if lines else None
    if detection is None:
        raise NoCannulaDetected(
            f"No cannula detected in this photo (detector confidence below "
            f"{threshold:.2f}). Retake it with the cannula hub and dressing "
            "clearly in frame."
        )

    x1, y1, x2, y2 = detection["bbox"]
    mx, my = (x2 - x1) * 0.4, (y2 - y1) * 0.4
    height, width = image.shape[:2]
    roi = [
        max(0.0, x1 - mx), max(0.0, y1 - my),
        min(float(width), x2 + mx), min(float(height), y2 + my),
    ]
    return roi, float(detection["confidence"]), [float(v) for v in detection["bbox"]]
