"""NIR-only vein scanning: the real provenance check, with only the model mocked."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core import modality


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _phone_jpeg() -> bytes:
    """A JPEG with camera EXIF, as a phone photo has."""
    image = Image.fromarray(np.full((64, 64, 3), 128, np.uint8))
    exif = Image.Exif()
    exif[modality.EXIF_MAKE] = "Apple"
    exif[modality.EXIF_MODEL] = "iPhone 15"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


@pytest.fixture
def model_calls(monkeypatch):
    """Record whether the vein model was reached; it must not be on a rejection."""
    from app.api import vein

    calls = []

    def fake(data):
        calls.append(data)
        return {
            "insertion_x": None, "insertion_y": None, "angle_deg": None,
            "vein_width_px": None, "confidence": 0.0, "reason": "empty mask",
            "mask_png": b"\x89PNG-fake", "overlay_png_base64": "ZmFrZQ==",
            "model_version": "attention-unet-test",
        }

    monkeypatch.setattr(vein, "run_vein_scan", fake)
    return calls


@pytest.fixture
def no_dataset(monkeypatch):
    monkeypatch.setattr(modality, "dataset_registry", lambda: {})


def _scan(client, patient, data, name="arm.png", ctype="image/png", headers=None):
    return client.post(
        f"/patients/{patient['id']}/vein-scan",
        files={"file": (name, data, ctype)}, headers=headers or {},
    )


class TestRejected:
    def test_phone_photo_rejected_as_rgb(self, client, patient, model_calls, no_dataset):
        response = _scan(client, patient, _phone_jpeg(), "IMG_0001.jpg", "image/jpeg")
        assert response.status_code == 422
        assert "RGB camera (Apple iPhone 15)" in response.json()["detail"]
        assert model_calls == []

    def test_plain_image_is_unverified(self, client, patient, model_calls, no_dataset):
        rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        response = _scan(client, patient, _png(rgb))
        assert response.status_code == 422
        assert "NIR vein scan required" in response.json()["detail"]
        assert "unverified" in response.json()["detail"]
        assert model_calls == []

    def test_grayscale_is_not_evidence_of_nir(self, client, patient, model_calls, no_dataset):
        gray = np.random.default_rng(1).integers(0, 255, (64, 64), dtype=np.uint8)
        assert _scan(client, patient, _png(gray)).status_code == 422
        assert model_calls == []

    def test_nir_filename_is_not_evidence_of_nir(self, client, patient, model_calls, no_dataset):
        gray = np.random.default_rng(2).integers(0, 255, (64, 64), dtype=np.uint8)
        assert _scan(client, patient, _png(gray), name="NIR_scan_850nm.png").status_code == 422
        assert model_calls == []

    def test_device_id_without_token_rejected(self, client, patient, model_calls, no_dataset, monkeypatch):
        monkeypatch.setattr(modality.get_settings(), "nir_devices", {"nir-cam-01": "s3cret"})
        response = _scan(client, patient, _png(np.zeros((64, 64), np.uint8)),
                         headers={"X-NIR-Device": "nir-cam-01"})
        assert response.status_code == 422
        assert model_calls == []

    def test_wrong_device_token_rejected(self, client, patient, model_calls, no_dataset, monkeypatch):
        monkeypatch.setattr(modality.get_settings(), "nir_devices", {"nir-cam-01": "s3cret"})
        response = _scan(client, patient, _png(np.zeros((64, 64), np.uint8)),
                         headers={"X-NIR-Device": "nir-cam-01", "X-NIR-Device-Token": "guess"})
        assert response.status_code == 422
        assert model_calls == []

    def test_corrupt_image_rejected(self, client, patient, model_calls):
        response = _scan(client, patient, b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert response.status_code == 422
        assert "not a readable image" in response.json()["detail"]
        assert model_calls == []

    def test_oversized_dimensions_rejected_without_decoding(self, client, patient, model_calls):
        huge = _png(np.zeros((6000, 8000), np.uint8))  # 48 MP, compresses to a few KB
        response = _scan(client, patient, huge)
        assert response.status_code == 413
        assert model_calls == []


class TestAccepted:
    def test_registered_device_with_token_accepted(self, client, patient, model_calls, no_dataset, monkeypatch):
        monkeypatch.setattr(modality.get_settings(), "nir_devices", {"nir-cam-01": "s3cret"})
        response = _scan(client, patient, _png(np.zeros((64, 64), np.uint8)),
                         headers={"X-NIR-Device": "nir-cam-01", "X-NIR-Device-Token": "s3cret"})
        assert response.status_code == 201
        body = response.json()
        assert body["modality_status"] == "verified"
        assert body["acquisition_source"] == "registered_device"
        assert body["inference_ms"] is not None
        assert len(model_calls) == 1

    def test_dataset_frame_accepted_with_its_split(self, client, patient, model_calls, monkeypatch):
        frame = _png(np.full((64, 64, 3), 90, np.uint8))
        import hashlib

        monkeypatch.setattr(modality, "dataset_registry", lambda: {
            hashlib.sha256(frame).hexdigest(): {"frame": "7", "split": "train"},
        })
        body = _scan(client, patient, frame).json()
        assert body["acquisition_source"] == "cubital_dataset"
        assert body["dataset_split"] == "train"

    def test_real_cubital_frame_is_in_registry(self):
        """Against the actual local dataset; skipped where CUBITAL is not downloaded."""
        path = modality.get_settings().repo_root / "data/raw/cubital/square_dataset/nir_images/7.png"
        if not path.exists():
            pytest.skip("CUBITAL not present")
        result = modality.verify_nir(path.read_bytes())
        assert result["source"] == "cubital_dataset"
        assert result["dataset_split"] == "train"
