import json
import subprocess
import sys

from PIL import Image

from metroyolo.domain import Detection, InferenceConfig
from metroyolo.evaluation import evaluate, save_evaluation


class StubDetector:
    def predict(self, image, config):
        return [Detection(0, 0, 10, 10, 0.7)]


def test_evaluation_writes_real_per_image_metrics(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (30, 30)).save(images / "a.png")
    (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n0 0.2 0.2 0.2 0.2")
    rows, sha = evaluate(StubDetector(), images, labels, InferenceConfig())
    output = tmp_path / "out"
    result = save_evaluation(output, rows, InferenceConfig(), sha, None)
    assert result["summary"]["count_mae"] == 1
    assert result["summary"]["occupancy_mae_pp"] == 5
    assert len(result["dataset_sha256"]) == 64
    assert "absolute_count_error" in (output / "per_image.csv").read_text()


def test_demo_cli_exports_image_and_explicit_provenance(tmp_path):
    command = [sys.executable, "-m", "metroyolo.cli", "demo", "--output", str(tmp_path)]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    data = json.loads((tmp_path / "prediction.json").read_text())
    assert data["source"] == "illustrative_fixture_not_ml"
    assert data["count"] == 8
    assert data["visible_area_load_percent"] == 40
    with Image.open(tmp_path / "annotated.png") as image:
        assert image.size == (1280, 720)


def test_missing_weights_yields_actionable_error(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "metroyolo.cli",
            "predict",
            "missing.png",
            "--repo",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "setup_yolov5.sh" in proc.stdout


def test_adapter_changes_settings_and_filters_other_classes(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from metroyolo.detector import YoloV5Detector

    class Tensor:
        def cpu(self):
            return self

        def tolist(self):
            return [[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 1]]

    class Model:
        def __call__(self, image, size):
            assert size == 640
            return SimpleNamespace(xyxy=[Tensor()])

    model = Model()

    def load(*args, **kwargs):
        assert kwargs["source"] == "local"
        return model

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(hub=SimpleNamespace(load=load), inference_mode=nullcontext),
    )
    (tmp_path / "hubconf.py").touch()
    weights = tmp_path / "trusted.pt"
    weights.touch()
    detector = YoloV5Detector(tmp_path, weights)
    detections = detector.predict(None, InferenceConfig(0.4, 0.3, 20))
    assert len(detections) == 1
    assert (model.conf, model.iou, model.max_det, model.classes) == (0.4, 0.3, 20, [0])


def test_sweep_ranks_by_count_mae_and_recomputes_each_config(monkeypatch, tmp_path):
    from metroyolo import detector
    from metroyolo.cli import parser, run

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (30, 30)).save(images / "a.png")
    (labels / "a.txt").write_text("0 0.5 0.5 0.5 0.5")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test-only fixture")
    calls = []

    class ConfigurableDetector:
        def __init__(self, *args):
            pass

        def predict(self, image, config):
            calls.append(config.confidence)
            return [Detection(0, 0, 10, 10, 0.9)] * (2 if config.confidence < 0.4 else 1)

    monkeypatch.setattr(detector, "YoloV5Detector", ConfigurableDetector)
    output = tmp_path / "sweep"
    args = parser().parse_args(
        [
            "sweep",
            "--images",
            str(images),
            "--labels",
            str(labels),
            "--weights",
            str(weights),
            "--output",
            str(output),
            "--confidences",
            "0.25",
            "0.55",
            "--ious",
            "0.45",
            "--max-dets",
            "20",
        ]
    )
    result = run(args)
    assert calls == [0.25, 0.55]
    assert result["best_on_validation"]["config"]["confidence"] == 0.55
    assert result["best_on_validation"]["summary"]["count_mae"] == 0
    assert len(json.loads((output / "ranking.json").read_text())) == 2


def test_evaluation_does_not_treat_missing_annotations_as_background(tmp_path):
    import pytest

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (30, 30)).save(images / "a.png")
    with pytest.raises(ValueError, match="Missing label"):
        evaluate(StubDetector(), images, tmp_path / "labels", InferenceConfig())


def test_model_load_failure_is_explained(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    from metroyolo.detector import YoloV5Detector

    def broken_load(*args, **kwargs):
        raise Exception("incompatible checkpoint")

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(hub=SimpleNamespace(load=broken_load))
    )
    (tmp_path / "hubconf.py").touch()
    weights = tmp_path / "trusted.pt"
    weights.touch()
    with pytest.raises(RuntimeError, match="model loading failed: incompatible checkpoint"):
        YoloV5Detector(tmp_path, weights)
