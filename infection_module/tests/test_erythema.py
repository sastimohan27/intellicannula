"""Functional tests for the peri-cannula redness measurement.

These check the measurement responds to redness the way it should. They do
not, and cannot, show that it predicts phlebitis - that needs nurse-scored
IV-site photos.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from infection_module.src import erythema

SKIN_BGR = (130, 160, 205)          # a mid-tone skin colour inside the YCrCb skin range
HUB = [280.0, 280.0, 320.0, 310.0]  # 40 x 30 px hub in a 600 x 600 photo


def _photo(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((600, 600, 3), SKIN_BGR, np.float32)
    img += rng.normal(0, 3, img.shape)                     # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = (int(v) for v in HUB)
    img[y1:y2, x1:x2] = (200, 120, 30)                     # blue plastic hub
    return img


def _redden(img: np.ndarray, amount: float, sigma: float = 60.0) -> np.ndarray:
    """Add `amount` of a* in a Gaussian around the hub (spreading redness)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx, cy = (HUB[0] + HUB[2]) / 2, (HUB[1] + HUB[3]) / 2
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    lab[..., 1] += amount * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def test_clean_site_reads_no_redness():
    result = erythema.measure(_photo(), HUB)
    assert result["reddened_fraction"] < 0.02
    assert result["redness_level"] == 0
    assert erythema.risk_score(result["reddened_fraction"]) < 0.05


def test_score_rises_monotonically_with_added_redness():
    fractions = [
        erythema.measure(_redden(_photo(), amount), HUB)["reddened_fraction"]
        for amount in (0, 6, 10, 16, 24)
    ]
    assert fractions == sorted(fractions)
    assert fractions[-1] > 0.4
    assert fractions[-1] - fractions[0] > 0.3


def test_uniform_skin_tone_change_is_not_redness():
    """A redder overall tone (lighting, skin colour) shifts the reference too."""
    shifted = _redden(_photo(), 10, sigma=10_000)  # effectively uniform
    assert erythema.measure(shifted, HUB)["reddened_fraction"] < 0.05


def test_white_tape_is_excluded_from_skin():
    img = _photo()
    img[250:270, 200:400] = (245, 245, 245)       # tape strip across the ring
    mask = erythema.skin_mask(img)
    assert not mask[255:265, 210:390].any()
    assert erythema.measure(img, HUB)["reddened_fraction"] < 0.02


def test_hub_plastic_is_not_skin():
    x1, y1, x2, y2 = (int(v) for v in HUB)
    assert not erythema.skin_mask(_photo())[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2].any()


def test_refuses_when_dressing_covers_the_skin():
    img = _photo()
    img[150:450, 150:450] = (240, 240, 240)       # opaque dressing over the whole ring
    x1, y1, x2, y2 = (int(v) for v in HUB)
    img[y1:y2, x1:x2] = (200, 120, 30)
    with pytest.raises(ValueError, match="visible skin"):
        erythema.measure(img, HUB)


def test_overlay_is_a_png():
    import base64

    png = base64.b64decode(erythema.overlay_png_base64(erythema.measure(_photo(), HUB)))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# --- baseline-anchored change -------------------------------------------------

def _scene() -> np.ndarray:
    """Skin photo with a white sheet strip (neutral pixels) and a hub."""
    img = _photo()
    img[:80, :] = (200, 200, 200)    # light-grey sheet: a colour cast does not clip it
    return img


def test_identical_photos_show_no_change():
    base = _scene()
    assert erythema.change_vs_baseline(base, HUB, base.copy())["change"] == 0.0


def test_white_balance_shift_is_not_change_when_neutrals_present():
    base = _scene()
    warm = np.clip(base.astype(np.float32) * [0.88, 1.0, 1.12], 0, 255).astype(np.uint8)
    result = erythema.change_vs_baseline(base, HUB, warm)
    assert result["colour_normalised"]
    assert result["change"] < 0.02


def test_injected_skin_redness_is_measured():
    base = _scene()
    skin = erythema.skin_mask(base).astype(np.float32)
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] += 12 * skin        # +12 a* on every skin pixel
    red = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    assert erythema.change_vs_baseline(base, HUB, red)["change"] > 0.9


def test_unaligned_follow_up_is_refused():
    with pytest.raises(ValueError, match="aligned"):
        erythema.change_vs_baseline(_scene(), HUB, _scene()[:500])
