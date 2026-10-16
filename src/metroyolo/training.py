"""Repeat the recovered YOLOv5 protocol; new results are measured, never filled in."""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from metroyolo.dataset import audit_dataset
from metroyolo.domain import InferenceConfig
from metroyolo.evaluation import evaluate, save_evaluation
from metroyolo.metrics import summarize

REVISIONS = {"915bbf294bb74c859f0b41f1c23bc395014ea679", "70b964b6d5067fff621f724c85d0e39e6b4c8e4e"}


def training_command(
    repo, data, weights, output, epochs, batch, image_size, device, seed, workers=8
):
    return [
        sys.executable,
        str((repo / "train.py").resolve()),
        "--img",
        str(image_size),
        "--batch",
        str(batch),
        "--epochs",
        str(epochs),
        "--data",
        str(data.resolve()),
        "--weights",
        str(weights.resolve()),
        "--project",
        str((output / "train").resolve()),
        "--name",
        f"metro_yolov5s_{epochs}",
        "--device",
        device,
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--optimizer",
        "SGD",
        "--hyp",
        str((repo / "data/hyps/hyp.scratch-low.yaml").resolve()),
    ]


def run_training(command, logfile):
    env = dict(os.environ, WANDB_MODE="disabled", YOLOv5_AUTOINSTALL="false")
    with logfile.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"Training failed; inspect {logfile}")
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def evaluate_checkpoint(weights, args, destination, config, device):
    from metroyolo.detector import YoloV5Detector

    detector = YoloV5Detector(args.repo, weights, device)
    rows, sha = evaluate(
        detector,
        args.dataset / "valid/images",
        args.dataset / "valid/labels",
        config,
        args.capacity,
    )
    return save_evaluation(destination, rows, config, sha, weights), detector


def sweep_checkpoint(detector, weights, args, destination):
    destination.mkdir()
    ranked = []
    for i, (conf, iou, max_det) in enumerate(
        itertools.product(
            [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            [15, 20, 25, 30, 40],
        )
    ):
        config = InferenceConfig(conf, iou, max_det, args.image_size)
        rows, sha = evaluate(
            detector,
            args.dataset / "valid/images",
            args.dataset / "valid/labels",
            config,
            args.capacity,
        )
        save_evaluation(destination / f"run_{i:03}", rows, config, sha, weights)
        ranked.append({"config": asdict(config), "summary": summarize(rows)})
        print(f"Sweep {i + 1}/350", flush=True)
    ranked.sort(
        key=lambda r: (
            r["summary"]["count_mae"],
            r["summary"]["occupancy_mae_pp"],
            r["summary"]["max_count_error"],
        )
    )
    (destination / "ranking.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    return ranked[0]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("data/metro"))
    p.add_argument("--repo", type=Path, default=Path("vendor/yolov5"))
    p.add_argument("--weights", type=Path, default=Path("weights/yolov5s.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--capacity", type=int, default=20)
    p.add_argument("--also-10-epochs", action="store_true")
    p.add_argument("--full-sweep", action="store_true")
    p.add_argument("--output", type=Path)
    return p


def run(args):
    if min(args.epochs, args.batch, args.capacity) < 1 or args.workers < 0:
        raise ValueError("Epochs, batch, capacity must be positive; workers nonnegative")
    config = InferenceConfig(max_detections=1000, image_size=args.image_size)
    if not (args.repo / "train.py").is_file() or not args.weights.is_file():
        raise ValueError("Set up the YOLOv5 checkout and pretrained weights first")
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision not in REVISIONS:
        raise ValueError("Expected pinned v7.0 or recovered historical YOLOv5 commit")
    audit = audit_dataset(args.dataset)
    if not audit["valid"]:
        raise ValueError(json.dumps(audit["errors"]))
    if not (args.dataset / "data.yaml").is_file():
        raise ValueError("Import data with scripts/import_dataset.py first")
    import torch

    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    inference_device = "cpu" if device == "cpu" else f"cuda:{device.split(',')[0]}"
    output = args.output or Path("runs/reproduce") / datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    if output.exists():
        raise ValueError(f"Output exists: {output}; use a new directory")
    output.mkdir(parents=True)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metadata = {
        "seed": args.seed,
        "historical_seed_known": True,
        "epochs": args.epochs,
        "batch": args.batch,
        "image_size": args.image_size,
        "device": device,
        "workers": args.workers,
        "torch": torch.__version__,
        "yolov5_commit": revision,
        "baseline_configuration": asdict(config),
        "provenance": "new training run",
    }
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    baseline, detector = evaluate_checkpoint(
        args.weights, args, output / "baseline", config, inference_device
    )
    del detector
    comparison = [{"experiment": "pretrained", **baseline["summary"]}]
    for epochs in dict.fromkeys([args.epochs] + ([10] if args.also_10_epochs else [])):
        command = training_command(
            args.repo,
            args.dataset / "data.yaml",
            args.weights,
            output,
            epochs,
            args.batch,
            args.image_size,
            device,
            args.seed,
            args.workers,
        )
        (output / f"train_{epochs}_command.json").write_text(json.dumps(command), encoding="utf-8")
        run_training(command, output / f"train_{epochs}.log")
        best = output / "train" / f"metro_yolov5s_{epochs}" / "weights/best.pt"
        if not best.is_file():
            raise RuntimeError(f"Training finished without checkpoint: {best}")
        measured, detector = evaluate_checkpoint(
            best, args, output / f"finetuned_{epochs}", config, inference_device
        )
        comparison.append({"experiment": f"finetuned_{epochs}", **measured["summary"]})
        if args.full_sweep and epochs == args.epochs:
            winner = sweep_checkpoint(detector, best, args, output / "sweep")
            (output / "best_validation_setting.json").write_text(json.dumps(winner, indent=2))
        del detector
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(comparison, indent=2))
    print(f"Completed: {output.resolve()}")
    return output


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
