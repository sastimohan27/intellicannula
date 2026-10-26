"""NIR provenance check for the vein-scan workflow.

The vein model was trained on near-infrared frames (CUBITAL). An ordinary RGB
photo is a different imaging modality: veins under the skin are largely
invisible to it, and feeding one to the model produces confident-looking but
meaningless masks. So the vein-scan endpoint accepts an image only when its
**source** is verified - never because of how its pixels look.

Why not pixels: measured on 60 CUBITAL NIR frames and 72 RGB photos, the colour
spread of NIR frames (median 10.2, 95th pct 13.5 in CIELAB chroma SD) overlaps
RGB photos (5th pct 2.8, median 7.5). CUBITAL's NIR camera renders a purple
tint, so NIR is not even "grayscale". Any pixel rule would be a guess.

Accepted sources:

* ``cubital_dataset`` - the upload's SHA-256 equals a frame of the local CUBITAL
  NIR dataset. Verifiable, and reports which split the frame belongs to so a
  demo on a training image is labelled as such.
* ``registered_device`` - the request carries ``X-NIR-Device`` and
  ``X-NIR-Device-Token`` matching a device registered server-side in
  ``settings.nir_devices`` (env ``NIR_DEVICES``, JSON). The token is the trust
  boundary: a device ID alone is user-editable and proves nothing.

Rejected:

* ``rgb_camera`` - the file carries camera EXIF (Make/Model), as phone and
  consumer-camera photos do, and is not from a registered NIR device.
* ``unverified`` - everything else. Filename, extension, grayscale appearance
  or a client-side label never establish NIR.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import logging
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

log = logging.getLogger(__name__)

NIR_REQUIRED_MESSAGE = (
    "NIR vein scan required. This image cannot be verified as a compatible "
    "near-infrared scan. Please capture or upload an image from a supported "
    "NIR imaging device."
)

EXIF_MAKE, EXIF_MODEL = 0x010F, 0x0110


class NIRNotVerified(ValueError):
    """The image's NIR provenance could not be established; `reason` says why."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@lru_cache(maxsize=1)
def dataset_registry() -> dict[str, dict]:
    """SHA-256 -> {"frame": id, "split": ...} for every local CUBITAL NIR frame.

    Built once per process (2,016 raw frames + processed split copies, under a
    second). Raw frames are mapped to their split through the processed
    `cubital_<id>.png` files, so provenance includes train/val/test.
    """
    root = get_settings().repo_root
    raw_dir = root / "data/raw/cubital/square_dataset/nir_images"
    processed = root / "data/processed/vein"

    split_of: dict[str, str] = {}
    registry: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        for path in (processed / split / "images").glob("cubital_*.png"):
            frame = path.stem.removeprefix("cubital_")
            split_of[frame] = split
            registry[_sha256(path)] = {"frame": frame, "split": split}

    for path in raw_dir.glob("*.png"):
        registry.setdefault(_sha256(path), {"frame": path.stem, "split": split_of.get(path.stem)})

    log.info("NIR dataset registry: %d frames", len(registry))
    return registry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera_exif(data: bytes) -> str | None:
    """'Make Model' from the file's EXIF, or None when absent/unreadable."""
    from PIL import Image

    try:
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:  # noqa: BLE001 - undecodable files are rejected elsewhere
        return None
    camera = " ".join(str(exif.get(tag, "")).strip() for tag in (EXIF_MAKE, EXIF_MODEL)).strip()
    return camera or None


def verify_nir(data: bytes, device_id: str | None = None, device_token: str | None = None) -> dict:
    """Establish NIR provenance or raise NIRNotVerified. Never inspects pixel statistics."""
    devices = get_settings().nir_devices
    if device_id:
        expected = devices.get(device_id)
        if expected is None or not device_token or not hmac.compare_digest(expected, device_token):
            raise NIRNotVerified(
                "unverified",
                f"{NIR_REQUIRED_MESSAGE} (NIR device {device_id!r} is not registered "
                "or its token is invalid.)",
            )
        return {"status": "verified", "source": "registered_device", "device_id": device_id,
                "dataset_split": None}

    match = dataset_registry().get(hashlib.sha256(data).hexdigest())
    if match is not None:
        return {"status": "verified", "source": "cubital_dataset", "device_id": None,
                "dataset_split": match["split"], "dataset_frame": match["frame"]}

    camera = _camera_exif(data)
    if camera:
        raise NIRNotVerified(
            "rgb_camera",
            f"This image is not accepted for vein segmentation: it was taken with an "
            f"RGB camera ({camera}). Phone cameras do not capture usable NIR vein images. "
            "Use a supported NIR imaging device.",
        )
    raise NIRNotVerified("unverified", f"{NIR_REQUIRED_MESSAGE} (NIR compatibility unverified.)")
