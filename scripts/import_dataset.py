"""Import the received Drive/Roboflow ZIP without redistributing its image contents."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from metroyolo.dataset import audit_dataset


def import_archive(archive: Path, destination: Path):
    if destination.exists():
        raise ValueError(f"Destination exists: {destination}; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        with ZipFile(archive) as source:
            for name in source.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Archive contains an unsafe path")
            for item in source.infolist():
                parts = Path(item.filename).parts
                if len(parts) != 4 or parts[0] != "dataset":
                    continue
                _, split, kind, filename = parts
                if split not in ("train", "valid") or kind not in ("images", "labels"):
                    continue
                suffix = Path(filename).suffix.lower()
                if kind == "labels" and suffix != ".txt":
                    continue
                if kind == "images" and suffix not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                target = staging / split / kind / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"Duplicate archive entry: {item.filename}")
                with source.open(item) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        report = audit_dataset(staging)
        if not report["valid"]:
            raise ValueError(json.dumps(report["errors"], ensure_ascii=False))
        yaml = (
            f"path: {json.dumps(str(destination.resolve()))}\n"
            "train: train/images\nval: valid/images\nnc: 1\nnames: ['person']\n"
        )
        (staging / "data.yaml").write_text(yaml, encoding="utf-8")
        report["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        (staging / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/metro"))
    args = parser.parse_args()
    try:
        report = import_archive(args.archive, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
