"""Validated values shared by inference, CLI and the dashboard."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    def __post_init__(self):
        values = (self.x1, self.y1, self.x2, self.y2, self.confidence)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Detection values must be finite")
        if self.x1 < 0 or self.y1 < 0 or self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Invalid detection rectangle")
        if not 0 <= self.confidence <= 1 or self.class_id < 0:
            raise ValueError("Invalid detection confidence/class")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class InferenceConfig:
    confidence: float = 0.25
    iou: float = 0.45
    max_detections: int = 100
    image_size: int = 640

    def __post_init__(self):
        if not 0 <= self.confidence <= 1 or not 0 < self.iou <= 1:
            raise ValueError("confidence must be in [0,1]; IoU must be in (0,1]")
        if self.max_detections < 1 or self.image_size < 32 or self.image_size % 32:
            raise ValueError("max_detections must be positive; image_size a multiple of 32")


def occupancy(count: int, capacity: int = 20) -> float:
    """Unclipped count/capacity indicator. Capacity describes the visible area."""
    if count < 0 or capacity <= 0:
        raise ValueError("count must be nonnegative; capacity must be positive")
    return 100 * count / capacity
