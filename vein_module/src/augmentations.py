"""Augmentation pipelines for the vein segmentation dataset.

Training-split augmentation only. Val/test get resize + normalise so that
metrics stay comparable across runs.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np

IMAGE_SIZE = 512


class CLAHETransform(A.ImageOnlyTransform):
    """CLAHE applied to the L channel in LAB space.

    Albumentations ships A.CLAHE, but it only accepts uint8 RGB and converts via
    its own path. This wrapper is explicit about operating on L in LAB so the
    behaviour matches `preprocessing.preprocess_image` exactly - the training
    and inference pipelines must apply identical contrast enhancement or the
    model sees a different distribution at deploy time.

    Handles single-channel (NIR) input by applying CLAHE directly, since NIR
    frames have no chroma to preserve.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        if img.ndim == 2:
            return clahe.apply(img)
        if img.shape[2] == 1:
            return clahe.apply(img[:, :, 0])[:, :, None]

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("clip_limit", "tile_grid_size")


def get_train_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Augmentation pipeline for the training split.

    Geometric ops are mild on purpose: forearm images are captured in a roughly
    canonical orientation, so large rotations would teach poses the model never
    meets. Vertical flip is excluded for the same reason - it would put the
    hand where the elbow belongs.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ]
    )


def get_eval_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Deterministic pipeline for val / test / inference.

    CLAHE stays on - it is preprocessing, not augmentation.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ]
    )
