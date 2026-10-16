"""Command-line interface for inference, data auditing and evaluation."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageOps

from .dataset import audit_dataset
from .domain import InferenceConfig, occupancy
from .evaluation import evaluate, save_evaluation
from .metrics import summarize


def parser():
    root = argparse.ArgumentParser(prog="metroyolo")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Export an illustrative example without ML dependencies")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    audit = sub.add_parser("audit", help="Validate YOLO labels and cross-split duplicate bytes")
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--output", type=Path)
    for name in ("predict", "evaluate", "sweep"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--repo", type=Path, default=Path("vendor/yolov5"))
        cmd.add_argument("--weights", type=Path, default=Path("weights/yolov5s.pt"))
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--confidence", type=float, default=0.25)
        cmd.add_argument("--iou", type=float, default=0.45)
        cmd.add_argument("--max-det", type=int, default=100)
        cmd.add_argument("--image-size", type=int, default=640)
        cmd.add_argument("--capacity", type=int, default=20)
        cmd.add_argument("--output", type=Path, default=Path(f"outputs/{name}"))
        if name == "predict":
            cmd.add_argument("image", type=Path)
        else:
            cmd.add_argument("--images", type=Path, required=True)
            cmd.add_argument("--labels", type=Path, required=True)
            cmd.add_argument("--allow-missing-labels", action="store_true")
        if name == "sweep":
            cmd.add_argument("--confidences", type=float, nargs="+", default=[0.25, 0.4, 0.55])
            cmd.add_argument("--ious", type=float, nargs="+", default=[0.3, 0.45])
            cmd.add_argument("--max-dets", type=int, nargs="+", default=[20, 40])
    return root


def export_prediction(image, detections, output, capacity, provenance):
    from .visualization import annotate

    output.mkdir(parents=True, exist_ok=True)
    annotate(image, detections).save(output / "annotated.png")
    result = {
        "source": provenance,
        "count": len(detections),
        "capacity": capacity,
        "visible_area_load_percent": occupancy(len(detections), capacity),
        "detections": [d.to_dict() for d in detections],
    }
    (output / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(args):
    if args.command == "demo":
        from .demo import demo_scene

        image, detections = demo_scene()
        return export_prediction(image, detections, args.output, 20, "illustrative_fixture_not_ml")
    if args.command == "audit":
        report = audit_dataset(args.dataset)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    config = InferenceConfig(args.confidence, args.iou, args.max_det, args.image_size)
    occupancy(0, args.capacity)
    from .detector import YoloV5Detector

    detector = YoloV5Detector(args.repo, args.weights, args.device)
    if args.command == "predict":
        with Image.open(args.image) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        detections = detector.predict(image, config)
        result = export_prediction(image, detections, args.output, args.capacity, "yolov5")
        result["config"] = asdict(config)
        (args.output / "prediction.json").write_text(json.dumps(result, indent=2))
        return result
    if args.command == "evaluate":
        rows, sha = evaluate(
            detector, args.images, args.labels, config, args.capacity, args.allow_missing_labels
        )
        return save_evaluation(args.output, rows, config, sha, args.weights)
    import itertools

    runs = []
    for index, (conf, iou, max_det) in enumerate(
        itertools.product(args.confidences, args.ious, args.max_dets)
    ):
        settings = InferenceConfig(conf, iou, max_det, args.image_size)
        rows, sha = evaluate(
            detector, args.images, args.labels, settings, args.capacity, args.allow_missing_labels
        )
        save_evaluation(args.output / f"run_{index:03d}", rows, settings, sha, args.weights)
        runs.append({"config": asdict(settings), "summary": summarize(rows)})
    runs.sort(key=lambda r: (r["summary"]["count_mae"], r["summary"]["max_count_error"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ranking.json").write_text(json.dumps(runs, indent=2))
    return {
        "n_configs": len(runs),
        "best_on_validation": runs[0],
        "note": "Selection on validation data; not an independent test estimate",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
        if isinstance(result, int):
            return result
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
