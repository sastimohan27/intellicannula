"""Tests for vein dataset preparation.

The subject-level split is the thing most worth guarding: an image-level split
leaks a subject's other arm into the test set and quietly inflates Dice.
"""

import json

import cv2
import numpy as np
import pytest

from vein_module.src.data_loader import (
    _binarize,
    _split_subjects,
    parse_subject_id,
    prepare_vein_dataset,
)


class TestParseSubjectId:
    def test_extracts_leading_number(self):
        assert parse_subject_id("0042_left.png", "cubital") == "cubital_0042"

    def test_handles_letter_prefix(self):
        assert parse_subject_id("s0042_R.png", "cubital") == "cubital_0042"

    def test_both_arms_of_one_subject_share_an_id(self):
        left = parse_subject_id("0042_left.png", "cubital")
        right = parse_subject_id("0042_right.png", "cubital")
        assert left == right

    def test_source_prefix_prevents_cross_dataset_collision(self):
        assert parse_subject_id("0042_a.png", "cubital") != parse_subject_id(
            "0042_a.png", "udavis"
        )

    def test_non_numeric_name_degrades_to_stem(self):
        assert parse_subject_id("odd-name.png", "udavis") == "udavis_odd-name"


class TestBinarize:
    def test_cubital_label_map_selects_vein_class_only(self):
        """0=background, 1=arm, 2=vein. Thresholding at 127 would return all zeros."""
        mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        out = _binarize(mask, "cubital")
        assert out.max() == 255
        assert out.tolist() == [[0, 0, 255], [255, 0, 0]]

    def test_cubital_scaled_encoding_also_handled(self):
        """Some CUBITAL versions ship the same map scaled to 0/128/255."""
        mask = np.array([[0, 128, 255]], dtype=np.uint8)
        out = _binarize(mask, "cubital")
        assert out.tolist() == [[0, 0, 255]]

    def test_udavis_uses_conventional_threshold(self):
        mask = np.array([[0, 200, 255]], dtype=np.uint8)
        out = _binarize(mask, "udavis")
        assert out.tolist() == [[0, 255, 255]]

    def test_output_is_strictly_binary(self):
        mask = np.array([[0, 1, 2, 2]], dtype=np.uint8)
        assert set(np.unique(_binarize(mask, "cubital"))) <= {0, 255}


class TestSplitSubjects:
    def test_every_subject_assigned_exactly_once(self):
        subjects = [f"s{i}" for i in range(100)]
        assignment = _split_subjects(subjects, 0.15, 0.15, seed=42)
        assert set(assignment) == set(subjects)
        assert set(assignment.values()) <= {"train", "val", "test"}

    def test_split_is_deterministic_for_a_seed(self):
        subjects = [f"s{i}" for i in range(50)]
        assert _split_subjects(subjects, 0.15, 0.15, 42) == _split_subjects(
            subjects, 0.15, 0.15, 42
        )

    def test_different_seeds_give_different_splits(self):
        subjects = [f"s{i}" for i in range(50)]
        assert _split_subjects(subjects, 0.15, 0.15, 1) != _split_subjects(
            subjects, 0.15, 0.15, 2
        )

    def test_fractions_roughly_respected(self):
        subjects = [f"s{i}" for i in range(100)]
        assignment = _split_subjects(subjects, 0.15, 0.15, seed=42)
        counts = {s: list(assignment.values()).count(s) for s in ("train", "val", "test")}
        assert counts["test"] == 15
        assert counts["val"] == 15
        assert counts["train"] == 70


@pytest.fixture
def fake_cubital(tmp_path):
    """Two images per subject, mimicking CUBITAL's left/right arm pairs."""
    root = tmp_path / "cubital"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)

    for subject in range(10):
        for arm in ("left", "right"):
            stem = f"{subject:04d}_{arm}"
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[20:30, :] = 2          # vein class
            mask[30:40, :] = 1          # arm class - must not survive binarisation
            cv2.imwrite(str(root / "images" / f"{stem}.png"), img)
            cv2.imwrite(str(root / "masks" / f"{stem}.png"), mask)
    return root


class TestPrepareVeinDataset:
    def test_no_subject_appears_in_two_splits(self, fake_cubital, tmp_path):
        out = tmp_path / "processed"
        prepare_vein_dataset(fake_cubital, tmp_path / "missing_udavis", out)

        manifest = json.loads((out / "split_manifest.json").read_text())
        train, val, test = (set(manifest["subjects"][s]) for s in ("train", "val", "test"))
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)

    def test_both_arms_land_in_the_same_split(self, fake_cubital, tmp_path):
        """The specific leak subject-level splitting exists to prevent."""
        out = tmp_path / "processed"
        prepare_vein_dataset(fake_cubital, tmp_path / "missing_udavis", out)

        location = {}
        for split in ("train", "val", "test"):
            for path in (out / split / "images").iterdir():
                subject = path.stem.split("_")[1]  # cubital_0003_left -> 0003
                location.setdefault(subject, set()).add(split)
        assert all(len(splits) == 1 for splits in location.values())

    def test_missing_udavis_is_tolerated(self, fake_cubital, tmp_path):
        """U-DAVIS has no public mirror; its absence must not be fatal."""
        counts = prepare_vein_dataset(
            fake_cubital, tmp_path / "does_not_exist", tmp_path / "processed"
        )
        assert sum(counts.values()) == 20

    def test_saved_masks_are_binary_and_drop_the_arm_class(self, fake_cubital, tmp_path):
        out = tmp_path / "processed"
        prepare_vein_dataset(fake_cubital, tmp_path / "missing", out)

        for split in ("train", "val", "test"):
            for path in (out / split / "masks").iterdir():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                assert set(np.unique(mask)) <= {0, 255}
                assert mask.max() == 255, "vein class was lost"

    def test_raises_when_no_data_present(self, tmp_path):
        with pytest.raises(RuntimeError, match="no samples found"):
            prepare_vein_dataset(tmp_path / "a", tmp_path / "b", tmp_path / "out")
