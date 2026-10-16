"""Shared evaluator; fresh inference for every NMS configuration."""

import csv
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from PIL import Image, ImageOps

from .dataset import fingerprint, images_in, read_labels
from .metrics import CountResult, summarize


def evaluate(detector, image_dir: Path, labels_dir: Path, config, capacity=20, allow_missing=False):
    images = images_in(image_dir)
    # Validate all ground truth before expensive inference or writing output.
    labels = [
        read_labels(labels_dir / f"{p.stem}.txt", allow_missing=allow_missing) for p in images
    ]
    results = []
    for path, boxes in zip(images, labels, strict=False):
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        detections = detector.predict(image, config)
        confidences = [d.confidence for d in detections]
        results.append(
            CountResult(
                path.name,
                len(boxes),
                len(detections),
                capacity,
                mean(confidences) if confidences else 0,
            )
        )
    return results, fingerprint(images, labels_dir)


def save_evaluation(destination: Path, results, config, dataset_sha, weights):
    destination.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in results]
    with (destination / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    weight_path = Path(weights) if weights else None
    import hashlib

    weight_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest() if weight_path else None
    payload = {
        "summary": summarize(results),
        "config": asdict(config),
        "dataset_sha256": dataset_sha,
        "weights_sha256": weight_sha,
        "weights": str(weights) if weights else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (destination / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
