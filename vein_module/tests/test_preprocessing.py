"""Tests for vein preprocessing.

Focus is on the invariants the training pipeline silently depends on: output
shape and dtype, and the failure modes that would otherwise produce an empty
Frangi channel without raising.
"""

import numpy as np
import pytest

from vein_module.src.preprocessing import (
    build_model_input,
    denoise,
    frangi_channel,
    preprocess_image,
)


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Synthetic forearm-like frame with a dark horizontal ridge."""
    img = np.full((64, 64, 3), 180, dtype=np.uint8)
    img[28:34, :, :] = 90  # a "vein": darker than surrounding tissue
    return img


@pytest.fixture
def nir_image() -> np.ndarray:
    img = np.full((64, 64), 180, dtype=np.uint8)
    img[28:34, :] = 90
    return img


class TestPreprocessImage:
    def test_rgb_preserves_shape_and_dtype(self, rgb_image):
        out = preprocess_image(rgb_image)
        assert out.shape == rgb_image.shape
        assert out.dtype == np.uint8

    def test_nir_single_channel_stays_2d(self, nir_image):
        out = preprocess_image(nir_image)
        assert out.shape == nir_image.shape
        assert out.ndim == 2

    def test_uniform_image_does_not_crash(self):
        flat = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = preprocess_image(flat)
        assert out.shape == flat.shape


class TestFrangiChannel:
    def test_detects_dark_ridge(self, rgb_image):
        out = frangi_channel(rgb_image, black_ridges=True)
        assert out.dtype == np.uint8
        assert out.shape == rgb_image.shape[:2]
        # The ridge rows should respond more strongly than the plain background.
        assert out[28:34, 20:44].mean() > out[0:6, 20:44].mean()

    def test_wrong_polarity_suppresses_response(self, rgb_image):
        """A dark-ridge image must not respond when asked for bright ridges.

        skimage's own `black_ridges` flag does not do this - it returns
        identical output either way in 0.25 - so this guards the explicit
        polarity mask in frangi_channel. If that mask is ever removed, this
        test fails rather than the model quietly learning from tendons and
        specular highlights.
        """
        correct = frangi_channel(rgb_image, black_ridges=True)
        flipped = frangi_channel(rgb_image, black_ridges=False)
        ridge = slice(28, 34), slice(20, 44)
        assert correct[ridge].mean() > flipped[ridge].mean()

    def test_bright_ridge_needs_black_ridges_false(self):
        """The mirror case: a bright vein on dark tissue."""
        img = np.full((64, 64, 3), 90, dtype=np.uint8)
        img[28:34, :, :] = 180
        ridge = slice(28, 34), slice(20, 44)
        assert (
            frangi_channel(img, black_ridges=False)[ridge].mean()
            > frangi_channel(img, black_ridges=True)[ridge].mean()
        )

    def test_response_magnitude_is_absolute_not_stretched(self):
        """A featureless frame must stay mostly dark.

        Min-max normalisation would stretch sensor noise to the full range and
        hand the network a confident channel built from nothing. Isolated
        spikes are expected - sigma=1 genuinely fires on single-pixel noise -
        so this checks the bulk of the frame, not its max.
        """
        noise = np.random.default_rng(0).integers(120, 130, (64, 64), dtype=np.uint8)
        assert frangi_channel(noise).mean() < 30

    def test_uniform_image_returns_zeros(self):
        """A blank frame has no ridges; normalising it would divide by zero."""
        flat = np.full((32, 32), 100, dtype=np.uint8)
        out = frangi_channel(flat)
        assert out.shape == (32, 32)
        assert out.max() == 0

    def test_all_black_image_returns_zeros(self):
        out = frangi_channel(np.zeros((32, 32), dtype=np.uint8))
        assert out.max() == 0


class TestBuildModelInput:
    def test_rgb_produces_three_channels(self, rgb_image):
        out = build_model_input(rgb_image)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_nir_produces_same_shape_as_rgb(self, nir_image, rgb_image):
        """One tensor shape across modalities, so one architecture serves both."""
        assert build_model_input(nir_image).shape == build_model_input(rgb_image).shape

    def test_third_channel_runs_frangi_on_the_denoised_frame(self, rgb_image):
        """Vesselness comes from the denoised frame, not the raw or CLAHE'd one."""
        out = build_model_input(rgb_image)
        np.testing.assert_array_equal(out[:, :, 2], frangi_channel(denoise(rgb_image)))

    def test_vesselness_separates_vein_from_background_under_noise(self):
        """The measurement that drove splitting denoise() out of preprocess_image().

        A low-contrast vein in sensor noise must score far above flat skin. If
        CLAHE ever creeps back into the vesselness path this ratio collapses by
        roughly 5x, which is the regression this guards.
        """
        rng = np.random.default_rng(0)
        base = np.full((128, 128, 3), 170, dtype=np.int16)
        base[58:66, :, :] = 140
        noisy = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)

        channel = build_model_input(noisy)[:, :, 2]
        ridge = channel[58:66, 20:108].mean()
        background = channel[5:25, 20:108].mean()
        assert ridge / max(background, 1e-6) > 100

    def test_clahe_path_would_be_noticeably_worse(self):
        """Documents the trade-off rather than asserting it by assumption."""
        rng = np.random.default_rng(0)
        base = np.full((128, 128, 3), 170, dtype=np.int16)
        base[58:66, :, :] = 140
        noisy = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)

        def ratio(channel):
            return channel[58:66, 20:108].mean() / max(
                channel[5:25, 20:108].mean(), 1e-6
            )

        assert ratio(frangi_channel(denoise(noisy))) > ratio(
            frangi_channel(preprocess_image(noisy))
        )
