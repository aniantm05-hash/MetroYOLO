import io
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/import_dataset.py"


def make_archive(path, invalid=False):
    with ZipFile(path, "w") as archive:
        for split, color in [("train", "red"), ("valid", "blue")]:
            image = io.BytesIO()
            Image.new("RGB", (32, 32), color).save(image, format="PNG")
            archive.writestr(f"dataset/{split}/images/a.png", image.getvalue())
            label = "0 0.5 0.5 0.5 0.5" if not invalid else "bad label"
            archive.writestr(f"dataset/{split}/labels/a.txt", label)
        archive.writestr("dataset/data.yaml", "train: /content/drive/old/path")
        archive.writestr("dataset/train/labels.cache", b"ignored cache")


def test_import_preserves_splits_and_creates_portable_configuration(tmp_path):
    archive = tmp_path / "input.zip"
    make_archive(archive)
    output = tmp_path / "data"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(archive), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    yaml = (output / "data.yaml").read_text()
    assert str(output.resolve()) in yaml
    assert "names: ['person']" in yaml
    assert not (output / "train/labels.cache").exists()
    assert (output / "train/labels/a.txt").read_text() == "0 0.5 0.5 0.5 0.5"


def test_bad_dataset_does_not_leave_partial_import(tmp_path):
    archive = tmp_path / "input.zip"
    make_archive(archive, invalid=True)
    output = tmp_path / "data"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(archive), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert not output.exists()


def test_archive_traversal_is_rejected(tmp_path):
    archive = tmp_path / "input.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("../escaped.txt", "unsafe")
    output = tmp_path / "data"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(archive), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert not (tmp_path / "escaped.txt").exists()
