"""Small adapter for the original YOLOv5 v7.0 implementation."""

import os
from pathlib import Path
from threading import Lock

from .domain import Detection, InferenceConfig


class YoloV5Detector:
    def __init__(self, repo: Path, weights: Path, device: str = "cpu"):
        if not (repo / "hubconf.py").is_file():
            raise ValueError("YOLOv5 checkout is missing. Run scripts/setup_yolov5.sh first.")
        if not weights.is_file():
            raise ValueError(f"Weights do not exist: {weights}")
        os.environ["YOLOv5_AUTOINSTALL"] = "false"
        import torch

        try:
            self.model = torch.hub.load(
                str(repo.resolve()),
                "custom",
                path=str(weights.resolve()),
                source="local",
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLOv5 model loading failed: {exc}") from exc
        self.lock = Lock()

    def predict(self, image, config: InferenceConfig) -> list[Detection]:
        import torch

        # Settings and inference form one critical section for shared Streamlit resources.
        with self.lock, torch.inference_mode():
            self.model.conf = config.confidence
            self.model.iou = config.iou
            self.model.max_det = config.max_detections
            self.model.classes = [0]
            output = self.model(image, size=config.image_size).xyxy[0].cpu().tolist()
        return [Detection(*row[:5], class_id=int(row[5])) for row in output if int(row[5]) == 0]
