# Experiment record

## Evidence available

The supplied practice PDF is the historical source. Printed pages 19–20 describe
81 training and 21 validation images. Page 26 reports the baseline metrics. Pages
28–30 describe the 640 px, batch 16, 30 epoch fine-tuning setup and detection metrics.
Pages 31–34 describe postprocessing search and the decision to keep the baseline.

The report contains code excerpts and aggregate values. The image/YOLO annotation archive was supplied on 2026-10-13 and audited.
The original notebook with training logs was subsequently supplied.
The historical per-image CSV and fine-tuned binary checkpoint remain unavailable. None of those artifacts is reconstructed or fabricated here.
The report itself includes cover material and external images and is not bundled in
this public-facing repository.

## What the report supports

- Pretrained YOLOv5s was evaluated on 21 validation images.
- Mean ground-truth count: 7.57; mean predicted count: 7.19.
- Count MAE: 1.05; maximum absolute count error: 5.
- Capacity was assumed to be 20 for the visible area.
- Load error: 5.24 percentage points; average per-image detection confidence: 0.519.
- Fine-tuning for 30 epochs reported precision 0.766, recall 0.654,
  mAP50 0.790 and mAP50–95 0.482.
- The report concludes that fine-tuning did not improve applied errors. It does not
  provide exact fine-tuned count MAE values or a full sweep result table.
- A 10 epoch run is described as producing excessive boxes; no numeric benchmark is provided.

## Interpretation

For each image, `absolute_count_error = abs(pred_count - true_count)`.
Count MAE is the mean of these errors. Load MAE is the mean of
`100 * absolute_count_error / capacity`, expressed in percentage points. With constant
capacity it is a scaled version of the same counting error.

The historical report's values are rounded. Calculating `1.05 * 5` from an already rounded
MAE yields 5.25 rather than the reported 5.24; unrounded per-image values are needed to
check the precise relation. This rounding discrepancy is not silently corrected.

A detection confidence is a model score, not the probability that the final passenger
count is correct. The reported 0.519 is an average confidence, not a selected threshold.
The recovered source defaults are confidence 0.25, IoU 0.45 and max_det 1000.
The initial notebook evaluation does not explicitly override them. The restored training
comparison uses those defaults; the earlier baseline rerun used max_det 100.

A model can achieve reasonable mAP while making additional count errors. No baseline mAP
is given in the report, so this project does not claim that fine-tuning improved mAP
relative to the baseline.

## Threats to validity

1. Labels were initialized with the baseline model, then manually edited. Remaining
   teacher-like errors can favor that model in the evaluation.
2. Small validation data makes results sensitive to individual images and scene composition.
3. Random image splitting can share scenes across train/validation. Exact-byte duplicate
   detection is necessary but insufficient; scene-aware splitting is the next step.
4. Postprocessing settings were selected on validation, not an untouched test set.
5. Small data and annotation noise are plausible explanations for poorer results;
   the report does not establish causality through controlled ablations.

## Re-running a comparison when the original artifacts are available

1. Restore the dataset, retain the split manifest and independently review validation labels.
2. Audit it with `metroyolo audit`; represent reviewed negative images with empty `.txt` files.
3. Evaluate both checkpoints on exactly the same images, recording configuration and hashes.
4. Tune thresholds on validation; freeze the selected configurations.
5. Report both models on independent, scene-separated test data; publish per-image count errors.
6. Measure latency separately with device, library versions, image size and warmup stated.

The evaluation CLI writes the configuration, dataset fingerprint, weights SHA256 and
per-image results. The sweep currently reruns inference for every setting; it is transparent
but not optimized for large datasets. Capacity is recorded per image in the exported CSV.

## Software added for the portfolio

The Python package, Streamlit application, strict data audit, exporters, sweeps, tests and
CI were implemented while preparing this repository. They are separate from the historical
training experiments. CLI demo assets remain labeled software fixtures. The application
now displays the three unchanged historical report figures instead of synthetic demo boxes.

## Received data audit · 2026-10-13

The archive contains the reported 81/21 split, 679/159 person labels and 9/1 empty
label files. Validation ground truth averages 159/21 = 7.5714 people per image,
consistent with the rounded 7.57 in the report. This supports the correspondence
but does not prove that every label byte is unchanged from the historical run.
The archive fingerprint and per-file manifest are retained. Cache files and the
additional unsplit label directory are not used in evaluation.

## Baseline rerun · 2026-10-13

The local evaluation completed on the received 21-image validation split with
YOLOv5 v7.0, official pretrained weights, image size 640, confidence 0.25,
NMS IoU 0.45, max detections 100 and capacity 20. No threshold search was performed.

| Metric | New run | Historical report |
|---|---:|---:|
| Mean ground-truth count | 7.571429 | 7.57 |
| Mean predicted count | 7.190476 | 7.19 |
| Count MAE | 1.047619 | 1.05 |
| Maximum count error | 5 | 5 |
| Load MAE, percentage points | 5.238095 | 5.24 |
| Mean image confidence | 0.518975 | 0.519 |

All these aggregate values agree after rounding. The count bias is -0.380952,
and exact count accuracy is 10/21 (47.62%). Per-image errors and hashes are retained
in `experiments/baseline_rerun/`. Matching rounded aggregates corroborates the baseline
result; original version and configuration records remain unavailable. The new run
still uses the same validation labels and does not resolve their possible teacher bias.
Fine-tuned weights were absent, so no new baseline-vs-fine-tuned comparison is claimed.

## Source notebook recovered

The original `Untitled1.ipynb` confirms the training code and records revision
`70b964b6d5067fff621f724c85d0e39e6b4c8e4e`, PyTorch 2.10.0+cu128, Ultralytics 8.4.47,
Tesla T4, SGD, seed 0 and `hyp.scratch-low.yaml`. The saved log confirms 30 completed
epochs and the detection metrics reported in the PDF. The run is named `metro_yolov5s_e10`.

The subsequent evaluation cell loads that same path and prints `10EPOCH`, count MAE
28.6190476, maximum error 64 and load MAE 143.095238 pp. Execution counts are absent,
so the saved file cannot establish whether the checkpoint was overwritten or the cells
were edited between runs. No separate executed 10-epoch training command is present.
These values are retained as ambiguous historical evidence, not a confirmed independent
10-epoch result. New experiments use separate, explicit 30/10 epoch names and directories.

The 350-setting sweep code is present but the final ranked table is absent from saved
outputs. The restored notebook executes a new sweep when enabled. It uses the recovered
source defaults (confidence 0.25, IoU 0.45, max_det 1000) for initial model comparisons.
The previous baseline rerun used max_det 100 and reproduced the reported count aggregates.
The sweep is selected on validation and is not an independent test result.
