"""Applied metrics; occupancy MAE is measured in percentage points."""

from dataclasses import asdict, dataclass
from statistics import mean

from .domain import occupancy


@dataclass(frozen=True)
class CountResult:
    image: str
    true_count: int
    pred_count: int
    capacity: int = 20
    avg_confidence: float = 0.0

    def __post_init__(self):
        occupancy(self.true_count, self.capacity)
        occupancy(self.pred_count, self.capacity)
        if not 0 <= self.avg_confidence <= 1:
            raise ValueError("Average confidence must be in [0,1]")

    def to_dict(self):
        return {
            **asdict(self),
            "absolute_count_error": abs(self.pred_count - self.true_count),
            "occupancy_true_percent": occupancy(self.true_count, self.capacity),
            "occupancy_pred_percent": occupancy(self.pred_count, self.capacity),
            "occupancy_error_pp": 100 * abs(self.pred_count - self.true_count) / self.capacity,
        }


def summarize(results: list[CountResult]) -> dict:
    if not results:
        raise ValueError("Cannot evaluate an empty dataset")
    errors = [abs(r.pred_count - r.true_count) for r in results]
    return {
        "n_images": len(results),
        "mean_true_count": mean(r.true_count for r in results),
        "mean_pred_count": mean(r.pred_count for r in results),
        "count_mae": mean(errors),
        "count_bias": mean(r.pred_count - r.true_count for r in results),
        "max_count_error": max(errors),
        "exact_count_accuracy": mean(e == 0 for e in errors),
        "occupancy_mae_pp": mean(
            100 * e / r.capacity for e, r in zip(errors, results, strict=False)
        ),
        "mean_image_confidence": mean(r.avg_confidence for r in results),
    }
