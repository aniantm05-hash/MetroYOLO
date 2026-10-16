#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash scripts/train.sh /absolute/path/to/data.yaml' >&2
  exit 2
fi
if [ ! -f "$1" ]; then
  echo 'Dataset YAML not found' >&2
  exit 2
fi
WANDB_MODE=disabled YOLOv5_AUTOINSTALL=false python vendor/yolov5/train.py   --img 640 --batch 16 --epochs 30 --seed 0   --data "$1" --weights weights/yolov5s.pt   --project runs/train --name metro_yolov5s --device 0
