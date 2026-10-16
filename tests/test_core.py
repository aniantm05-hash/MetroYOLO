import math

import pytest

from metroyolo.domain import Detection, InferenceConfig, occupancy
from metroyolo.metrics import CountResult, summarize


@pytest.mark.parametrize("count,capacity,expected", [(0, 20, 0), (7, 20, 35), (25, 20, 125)])
def test_load_is_unclipped(count, capacity, expected):
    assert occupancy(count, capacity) == expected


@pytest.mark.parametrize("count,capacity", [(-1, 20), (1, 0), (1, -2)])
def test_load_invalid_inputs(count, capacity):
    with pytest.raises(ValueError):
        occupancy(count, capacity)


def test_metrics_are_task_specific():
    rows = [CountResult("a", 3, 5), CountResult("b", 5, 4), CountResult("c", 0, 0)]
    result = summarize(rows)
    assert result["count_mae"] == 1
    assert result["count_bias"] == pytest.approx(1 / 3)
    assert result["occupancy_mae_pp"] == 5
    assert result["max_count_error"] == 2
    assert result["exact_count_accuracy"] == pytest.approx(1 / 3)


def test_capacity_can_vary_per_image():
    result = summarize([CountResult("a", 0, 1, 10), CountResult("b", 0, 1, 20)])
    assert result["occupancy_mae_pp"] == 7.5


def test_empty_evaluation_is_rejected():
    with pytest.raises(ValueError):
        summarize([])


@pytest.mark.parametrize(
    "kwargs", [{"confidence": -0.1}, {"iou": 0}, {"max_detections": 0}, {"image_size": 641}]
)
def test_invalid_inference_settings(kwargs):
    with pytest.raises(ValueError):
        InferenceConfig(**kwargs)


@pytest.mark.parametrize(
    "box", [(0, 0, 0, 3, 0.5), (0, 0, 3, 3, 1.1), (0, 0, math.nan, 3, 0.5), (-1, 0, 3, 3, 0.5)]
)
def test_invalid_detection_values(box):
    with pytest.raises(ValueError):
        Detection(*box)
