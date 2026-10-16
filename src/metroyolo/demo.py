"""Fixed illustrative fixture, deliberately not a neural-network prediction."""

import json
from pathlib import Path

from .domain import Detection

ROOT = Path(__file__).resolve().parent


def demo_scene():
    # Bundled fixture also works in a built wheel.
    base = ROOT / "assets"
    from PIL import Image

    image = Image.open(base / "scene.png").convert("RGB")
    data = json.loads((base / "detections.json").read_text())
    return image, [Detection(**d) for d in data["detections"]]
