from pathlib import Path

import pytest
from PIL import Image

from metroyolo.dataset import audit_dataset, images_in, read_labels


@pytest.mark.parametrize(
    "line",
    [
        "1 0.5 0.5 0.2 0.2",
        "0 nan 0.5 0.2 0.2",
        "0 0.1 0.5 0.5 0.2",
        "0 0.5 0.5 0 0.2",
        "0 0.5 0.5 0.2",
        "x 0.5 0.5 0.2 0.2",
    ],
)
def test_bad_labels_rejected(tmp_path, line):
    label = tmp_path / "sample.txt"
    label.write_text(line)
    with pytest.raises(ValueError, match="sample.txt:1"):
        read_labels(label)


def test_missing_is_different_from_empty(tmp_path):
    label = tmp_path / "sample.txt"
    with pytest.raises(ValueError, match="Missing label"):
        read_labels(label)
    assert read_labels(label, allow_missing=True) == []
    label.write_text("")
    assert read_labels(label) == []


def create_split(root: Path, split: str, color: str):
    folder = root / split
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir()
    Image.new("RGB", (32, 32), color).save(folder / "images" / "a.png")
    (folder / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5")


def test_duplicate_across_splits_is_a_leak(tmp_path):
    create_split(tmp_path, "train", "red")
    create_split(tmp_path, "valid", "red")
    report = audit_dataset(tmp_path)
    assert not report["valid"]
    assert len(report["duplicate_images"]) == 1


def test_dataset_fingerprint_changes_with_labels(tmp_path):
    create_split(tmp_path, "train", "red")
    create_split(tmp_path, "valid", "blue")
    first = audit_dataset(tmp_path)
    assert first["valid"]
    (tmp_path / "train" / "labels" / "a.txt").write_text("")
    second = audit_dataset(tmp_path)
    assert first["splits"]["train"]["sha256"] != second["splits"]["train"]["sha256"]
    assert second["splits"]["train"]["verified_empty_labels"] == 1


def test_same_stem_is_ambiguous(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
    Image.new("RGB", (10, 10)).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="identical stems"):
        images_in(tmp_path)
