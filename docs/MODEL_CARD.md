# Model card · YOLOv5s passenger counting

| Item | Detail |
|---|---|
| Model | YOLOv5s pretrained on COCO; person class (ID 0) |
| Code revision | YOLOv5 v7.0, `915bbf294bb74c859f0b41f1c23bc395014ea679` |
| Task | Detect and count visible people in individual photographs |
| Input | RGB image; EXIF orientation applied; inference size defaults to 640 |
| Output | Person bounding boxes, confidence scores, count, assumed visible-area load index |
| Runtime | PyTorch 2.5.1, torchvision 0.20.1, CPU default; optional CUDA |
| Fine-tuned weights | Not included; use a locally supplied compatible `best.pt` |
| Historical evaluation | 21 validation images; transcribed report values, not a reproduced dataset run |

## Intended use

Educational research, portfolio demonstrations and analysis of user-provided still images.
The project does not estimate unseen passengers, total carriage occupancy or unique people
across a video. No tracking, camera calibration, deployment throughput or real-time guarantees
are implemented.

## Failure modes

Occlusion, dense crowds, reflections, partial bodies, unusual viewpoints, motion blur,
poor lighting and domain shift can cause missed people or false positives. A photograph
only covers part of the carriage. Counts and model confidence do not provide a calibrated
uncertainty interval. The capacity parameter is a user assumption.

## Evaluation limitations

The original dataset is small and partially annotated using the baseline model.
No independent test set or numeric count benchmark for fine-tuned weights was supplied.
Reported mAP values and the application's count MAE describe different quantities.

## Data and use considerations

Process only images you are authorized to use. The local dashboard has no identity
recognition or persistent upload store implemented. Loaded images live in the current process;
PNG/JSON outputs are written only when explicitly exported by the user or the CLI.
The first model setup downloads upstream code and weights. Only load trusted local checkpoints.
The repository contains no external passenger photographs or original training weights.

## New baseline evaluation · 2026-10-13

The received validation data was evaluated: count MAE 1.047619, visible-area load
MAE 5.238095 percentage points. All six rounded aggregate values match the report.
Configuration, input/weight hashes and per-image errors are available in
`experiments/baseline_rerun/`. Fine-tuned checkpoint evaluation remains unavailable.
