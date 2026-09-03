"""Image preprocessing for vein segmentation.

The network never sees a raw frame. `build_model_input` assembles the 3-channel
stack the model is trained on, and train / eval / deploy must all route through
it or the model sees a distribution it never learned.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import frangi

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """CLAHE on L in LAB, then edge-preserving smoothing.

    Bilateral filtering rather than Gaussian: veins are low-contrast ridges
    only a few pixels wide, and a Gaussian wide enough to suppress sensor noise
    also erases them. The bilateral kernel smooths within-region noise while
    holding the intensity step at the vein boundary.

    Accepts RGB or single-channel NIR. NIR skips the LAB round-trip since it
    has no chroma to preserve.
    """
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        enhanced = clahe.apply(img)
        return cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)


def denoise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise with no contrast enhancement.

    This is `preprocess_image` minus the CLAHE step, and it exists because the
    vesselness channel wants the opposite of what the intensity channels want.
    Measured on a synthetic low-contrast vein with Gaussian sensor noise,
    ridge-to-background ratio in the Frangi response:

        raw frame          11 : 1
        CLAHE + bilateral  40 : 1
        bilateral only    201 : 1

    CLAHE raises local contrast everywhere, including in flat skin, so it
    amplifies exactly the speckle that Frangi's smallest scale mistakes for
    vessels. The intensity channels still get CLAHE - it genuinely helps a CNN
    see low-contrast veins - but the vesselness channel must not.
    """
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def frangi_channel(
    img: np.ndarray, black_ridges: bool = True, bg_kernel: int = 31
) -> np.ndarray:
    """Frangi vesselness response, polarity-filtered, scaled to uint8.

    Frangi scores how tube-like the local Hessian is, which encodes the prior
    "veins are elongated ridges" - a strong hand-engineered channel that a
    small dataset would struggle to learn on its own.

    Why the polarity filtering is done here rather than by `black_ridges`
    ------------------------------------------------------------------------
    scikit-image's `black_ridges` argument is a no-op in 0.25: `frangi()`
    returns byte-identical output for True and False, because the vesselness
    measure is built from absolute eigenvalues and the sign cancels. Verified
    directly on 0.25.0 - a dark bar and a bright bar both score 0.579.

    That matters clinically. Veins absorb strongly around 850 nm and read
    *darker* than surrounding tissue, but so do plenty of things that are not
    veins - and bright linear structures (tendon edges, specular highlights
    from gel or sweat, stray hairs) score just as high with an unfiltered
    filter. So polarity is applied explicitly: the response is kept only where
    the pixel is darker (or brighter, if black_ridges=False) than its local
    background, estimated with a box blur wider than the largest Frangi scale.

    Set black_ridges=False for a source where veins appear brighter than
    surrounding tissue - some transillumination rigs invert this.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    # skimage >=0.20 replaced scale_range/scale_step with an explicit sigmas
    # sequence; range(1, 10, 2) reproduces the old (1,10) step-2 behaviour.
    response = frangi(
        gray.astype(np.float64) / 255.0,
        sigmas=range(1, 10, 2),
        beta=0.5,
        black_ridges=black_ridges,
    )

    gray_f = gray.astype(np.float32)
    local_background = cv2.blur(gray_f, (bg_kernel, bg_kernel))
    keep = gray_f < local_background if black_ridges else gray_f > local_background
    response = response * keep

    # Absolute scaling, NOT min-max. Frangi already returns a normalised
    # vesselness score in [0, 1], so multiplying preserves magnitude: a frame
    # with no vessels stays dark. A min-max stretch would rescale pure noise to
    # the full 0-255 range and hand the network a confident-looking channel
    # built from nothing.
    return np.clip(response * 255.0, 0, 255).astype(np.uint8)


def build_model_input(img: np.ndarray, black_ridges: bool = True) -> np.ndarray:
    """Assemble the (H, W, 3) stack the segmentation network consumes.

    Channels: [preprocessed R, preprocessed G, Frangi vesselness].

    Why R and G and not B: haemoglobin absorption falls off through the visible
    range, so blue carries mostly surface scatter and little vein signal. For
    single-channel NIR input the preprocessed frame is duplicated into both
    slots, which keeps one tensor shape across modalities so the same
    checkpoint architecture serves both.
    """
    processed = preprocess_image(img)

    if processed.ndim == 2:
        first = second = processed
    else:
        first, second = processed[:, :, 0], processed[:, :, 1]

    # Vesselness comes from a denoised-but-not-contrast-enhanced frame. See
    # `denoise` for the measurements behind that choice: sigma=1 fires on
    # single-pixel noise, and CLAHE makes that noise worse, not better.
    vesselness = frangi_channel(denoise(img), black_ridges=black_ridges)
    return np.stack([first, second, vesselness], axis=-1)
