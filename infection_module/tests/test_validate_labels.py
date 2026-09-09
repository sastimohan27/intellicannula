"""Tests for the cannula label validator.

Each test pins one failure mode that would otherwise waste a full labelling
session or, worse, silently produce an inflated test score.
"""

from __future__ import annotations

import pytest

from infection_module.src.cannula_detector.validate_labels import (
    _patient_id,
    validate,
    validate_split,
)


def _make_split(root, split, entries):
    """entries: {stem: label_text or None}. None means no label file at all."""
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for stem, text in entries.items():
        (images / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        if text is not None:
            (labels / f"{stem}.txt").write_text(text)


VALID = "0 0.5 0.5 0.2 0.15\n"


class TestPatientId:
    def test_extracts_prefix(self):
        assert _patient_id("p001_3") == "p001"

    def test_photos_of_one_patient_share_an_id(self):
        assert _patient_id("p001_1") == _patient_id("p001_2")


class TestLabelFormat:
    def test_valid_label_passes(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": VALID})
        issues, stats = validate_split(tmp_path, "train")
        assert [i for i in issues if i.level == "ERROR"] == []
        assert stats["boxes"] == 1

    def test_pixel_coordinates_rejected(self, tmp_path):
        """The most common labelling export mistake."""
        _make_split(tmp_path, "train", {"p1_1": "0 320 240 80 60\n"})
        issues, _ = validate_split(tmp_path, "train")
        assert any("outside [0,1]" in i.message for i in issues)

    def test_wrong_field_count_rejected(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": "0 0.5 0.5 0.2\n"})
        issues, _ = validate_split(tmp_path, "train")
        assert any("expected 5 fields" in i.message for i in issues)

    def test_non_zero_class_id_rejected(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": "1 0.5 0.5 0.2 0.15\n"})
        issues, _ = validate_split(tmp_path, "train")
        assert any("expected 0 (cannula_hub)" in i.message for i in issues)

    def test_non_numeric_rejected(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": "0 a b c d\n"})
        issues, _ = validate_split(tmp_path, "train")
        assert any("non-numeric" in i.message for i in issues)

    def test_zero_area_box_rejected(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": "0 0.5 0.5 0.0 0.1\n"})
        issues, _ = validate_split(tmp_path, "train")
        assert any("zero-area" in i.message for i in issues)

    def test_whole_frame_box_warns(self, tmp_path):
        """A box round the whole dressing defeats crop_roi's margin."""
        _make_split(tmp_path, "train", {"p1_1": "0 0.5 0.5 0.95 0.95\n"})
        issues, _ = validate_split(tmp_path, "train")
        assert any("tight box" in i.message for i in issues)


class TestPairing:
    def test_missing_label_is_an_error(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": None})
        issues, _ = validate_split(tmp_path, "train")
        assert any("has no label file" in i.message for i in issues)

    def test_empty_label_is_valid_and_counted_as_negative(self, tmp_path):
        """An explicit empty file means 'no cannula here', which is useful."""
        _make_split(tmp_path, "train", {"p1_1": ""})
        issues, stats = validate_split(tmp_path, "train")
        assert [i for i in issues if i.level == "ERROR"] == []
        assert stats["empty_labels"] == 1

    def test_orphan_label_is_an_error(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": VALID})
        (tmp_path / "train" / "labels" / "ghost.txt").write_text(VALID)
        issues, _ = validate_split(tmp_path, "train")
        assert any("no matching image" in i.message for i in issues)

    def test_missing_split_directory_is_an_error(self, tmp_path):
        issues, _ = validate_split(tmp_path, "train")
        assert any("missing directory" in i.message for i in issues)


class TestPatientLeakage:
    def test_same_patient_in_train_and_test_is_an_error(self, tmp_path):
        """The failure that inflated the AZH risk model to F1 0.99."""
        _make_split(tmp_path, "train", {"p1_1": VALID})
        _make_split(tmp_path, "val", {"p2_1": VALID})
        _make_split(tmp_path, "test", {"p1_2": VALID})
        issues, _ = validate(tmp_path)
        assert any("patient leakage" in i.message for i in issues)

    def test_disjoint_patients_pass(self, tmp_path):
        _make_split(tmp_path, "train", {"p1_1": VALID})
        _make_split(tmp_path, "val", {"p2_1": VALID})
        _make_split(tmp_path, "test", {"p3_1": VALID})
        issues, _ = validate(tmp_path)
        assert not any("leakage" in i.message for i in issues)


class TestDatasetLevelWarnings:
    def test_small_dataset_warns(self, tmp_path):
        for split in ("train", "val", "test"):
            _make_split(tmp_path, split, {f"p{split}_1": VALID})
        issues, _ = validate(tmp_path)
        assert any("Below ~100" in i.message for i in issues)

    def test_no_negatives_warns(self, tmp_path):
        """Without hard negatives the detector fires on a wristwatch."""
        for split in ("train", "val", "test"):
            _make_split(tmp_path, split, {f"p{split}_1": VALID})
        issues, _ = validate(tmp_path)
        assert any("no negative images" in i.message for i in issues)
