"""Tests for infection dataset preparation.

Two invariants matter most here:
  1. Medetec never leaks into the training splits - it is the external
     generalization test, and mixing it in would make that number meaningless.
  2. The proxy label mapping is exactly wound=1 / normal=0 / background dropped,
     because every Phase 5 metric is reported against it.
"""

import json

import cv2
import numpy as np
import pytest

from infection_module.src.data_loader import (
    HEALTHY_CLASSES,
    WOUND_CLASSES,
    WoundClassificationDataset,
    _stratified_split,
    prepare_infection_dataset,
)


def _write_images(directory, n, prefix="img"):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(directory / f"{prefix}_{i:03d}.png"), img)


@pytest.fixture
def fake_azh(tmp_path):
    """AZH's real layout: Train/ and Test/ trees with single-letter class dirs."""
    root = tmp_path / "azh"
    for split, scale in (("Train", 1), ("Test", 0)):
        for cls, n in (("D", 20), ("P", 20), ("V", 20), ("S", 20), ("N", 20), ("BG", 20)):
            _write_images(root / "dataset" / split / cls, n if scale else 4, prefix=f"{split}{cls}")
    return root


@pytest.fixture
def fake_medetec(tmp_path):
    root = tmp_path / "medetec"
    _write_images(root / "slide scans" / "extravasation", 10, prefix="extrav")
    _write_images(root / "slide scans" / "burns", 8, prefix="burn")
    return root


class TestStratifiedSplit:
    def test_all_images_accounted_for(self):
        paths = [f"img{i}.png" for i in range(100)]
        splits = _stratified_split(paths, 0.15, 0.15, __import__("random").Random(42))
        assert sum(len(v) for v in splits.values()) == 100

    def test_no_image_in_two_splits(self):
        paths = [f"img{i}.png" for i in range(100)]
        splits = _stratified_split(paths, 0.15, 0.15, __import__("random").Random(42))
        assert not (set(splits["train"]) & set(splits["val"]))
        assert not (set(splits["train"]) & set(splits["test"]))
        assert not (set(splits["val"]) & set(splits["test"]))

    def test_fractions_respected(self):
        paths = [f"img{i}.png" for i in range(100)]
        splits = _stratified_split(paths, 0.15, 0.15, __import__("random").Random(42))
        assert len(splits["test"]) == 15
        assert len(splits["val"]) == 15
        assert len(splits["train"]) == 70


class TestPrepareInfectionDataset:
    def test_background_class_is_dropped(self, fake_azh, fake_medetec, tmp_path):
        out = tmp_path / "processed"
        counts = prepare_infection_dataset(fake_azh, fake_medetec, out)
        for split in ("train", "val", "test"):
            assert "background" not in counts[split]

    def test_medetec_kept_out_of_training_splits(self, fake_azh, fake_medetec, tmp_path):
        """Mixing the external test set into train would invalidate transfer numbers."""
        out = tmp_path / "processed"
        prepare_infection_dataset(fake_azh, fake_medetec, out)

        external = {p.name for p in (out / "medetec_external_test").rglob("*.png")}
        assert external, "external test set is empty"
        for split in ("train", "val", "test"):
            internal = {p.name for p in (out / split).rglob("*.png")}
            assert not (internal & external)

    def test_extravasation_category_survives(self, fake_azh, fake_medetec, tmp_path):
        """Extravasation is the closest public analogue to an IV-site injury."""
        out = tmp_path / "processed"
        counts = prepare_infection_dataset(fake_azh, fake_medetec, out)
        assert "extravasation" in counts["medetec_external_test"]

    def test_stratified_across_classes(self, fake_azh, fake_medetec, tmp_path):
        out = tmp_path / "processed"
        counts = prepare_infection_dataset(fake_azh, fake_medetec, out)
        for cls in ("diabetic", "pressure", "venous", "surgical", "normal"):
            assert counts["train"][cls] > 0
            assert counts["test"][cls] > 0

    def test_manifest_records_proxy_label_warning(self, fake_azh, fake_medetec, tmp_path):
        """The proxy caveat must travel with the data, not just live in a docstring."""
        out = tmp_path / "processed"
        prepare_infection_dataset(fake_azh, fake_medetec, out)
        manifest = json.loads((out / "split_manifest.json").read_text())
        assert "PROXY" in manifest["label_provenance"]
        assert manifest["split_level"] == "image"

    def test_missing_medetec_is_tolerated(self, fake_azh, tmp_path):
        counts = prepare_infection_dataset(fake_azh, tmp_path / "nope", tmp_path / "out")
        assert counts["train"]

    def test_raises_when_azh_absent(self, tmp_path):
        with pytest.raises(RuntimeError, match="no AZH images"):
            prepare_infection_dataset(tmp_path / "a", tmp_path / "b", tmp_path / "out")


class TestWoundClassificationDataset:
    def test_binary_labels_follow_the_proxy_mapping(self, fake_azh, fake_medetec, tmp_path):
        out = tmp_path / "processed"
        prepare_infection_dataset(fake_azh, fake_medetec, out)

        ds = WoundClassificationDataset(out, "train", label_mode="binary")
        for path, label in ds.samples:
            expected = 1 if path.parent.name in WOUND_CLASSES else 0
            assert label == expected
            assert path.parent.name in WOUND_CLASSES | HEALTHY_CLASSES

    def test_type_mode_gives_distinct_class_indices(self, fake_azh, fake_medetec, tmp_path):
        out = tmp_path / "processed"
        prepare_infection_dataset(fake_azh, fake_medetec, out)

        ds = WoundClassificationDataset(out, "train", label_mode="type")
        assert len(set(ds.class_to_idx.values())) == len(ds.classes)
        assert len(ds.classes) == 5  # 4 wound types + normal, background dropped

    def test_getitem_returns_chw_float_in_unit_range(self, fake_azh, fake_medetec, tmp_path):
        out = tmp_path / "processed"
        prepare_infection_dataset(fake_azh, fake_medetec, out)

        image, label = WoundClassificationDataset(out, "train", image_size=64)[0]
        assert image.shape == (3, 64, 64)
        assert image.dtype == np.float32
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert label in (0, 1)

    def test_rejects_unknown_label_mode(self, fake_azh, fake_medetec, tmp_path):
        out = tmp_path / "processed"
        prepare_infection_dataset(fake_azh, fake_medetec, out)
        with pytest.raises(ValueError, match="label_mode"):
            WoundClassificationDataset(out, "train", label_mode="nonsense")

    def test_missing_split_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--prepare"):
            WoundClassificationDataset(tmp_path, "train")
