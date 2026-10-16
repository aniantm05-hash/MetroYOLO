"""YOLO label validation and dataset audit. Missing labels are not silently negatives."""

import hashlib
import math
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Image directory does not exist: {folder}")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images in {folder}")
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Different images have identical stems and share a label filename")
    return images


def read_labels(path: Path, *, allow_missing: bool = False) -> list[tuple]:
    if not path.exists():
        if allow_missing:
            return []
        raise ValueError(f"Missing label: {path}; empty .txt files mark verified negatives")
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("Expected class xc yc width height")
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            if cls != 0 or not all(math.isfinite(v) for v in (xc, yc, w, h)):
                raise ValueError("Only finite person (class 0) labels are supported")
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("YOLO coordinates must be normalized and nonempty")
            if (
                xc - w / 2 < -1e-6
                or yc - h / 2 < -1e-6
                or xc + w / 2 > 1 + 1e-6
                or yc + h / 2 > 1 + 1e-6
            ):
                raise ValueError("Box crosses image boundaries")
            boxes.append((cls, xc, yc, w, h))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return boxes


def fingerprint(images: list[Path], labels: Path) -> str:
    digest = hashlib.sha256()
    for image in images:
        label = labels / f"{image.stem}.txt"
        for name, payload in (
            (image.name, image.read_bytes()),
            (label.name, label.read_bytes() if label.exists() else b"<missing>"),
        ):
            digest.update(name.encode() + b"\0" + hashlib.sha256(payload).digest())
    return digest.hexdigest()


def audit_dataset(root: Path) -> dict:
    report = {"splits": {}, "errors": [], "warnings": [], "duplicate_images": []}
    hashes = {}
    for split in ("train", "valid"):
        image_dir, label_dir = root / split / "images", root / split / "labels"
        try:
            images = images_in(image_dir)
        except ValueError as exc:
            report["errors"].append(str(exc))
            continue
        negatives, objects = 0, 0
        for image in images:
            try:
                with Image.open(image) as im:
                    im.verify()
                boxes = read_labels(label_dir / f"{image.stem}.txt")
                objects += len(boxes)
                negatives += not boxes
            except (ValueError, OSError) as exc:
                report["errors"].append(str(exc))
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                other_split, other_image = hashes[digest]
                report["duplicate_images"].append([str(other_image), str(image)])
                severity = "errors" if other_split != split else "warnings"
                report[severity].append(f"Duplicate bytes: {other_image} and {image}")
            else:
                hashes[digest] = (split, image)
        known = {p.stem for p in images}
        orphans = [p.name for p in label_dir.glob("*.txt") if p.stem not in known]
        if orphans:
            report["warnings"].append(f"{split}: orphan labels {orphans}")
        report["splits"][split] = {
            "images": len(images),
            "people": objects,
            "verified_empty_labels": negatives,
            "sha256": fingerprint(images, label_dir),
        }
    report["valid"] = not report["errors"]
    return report
