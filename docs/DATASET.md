# Dataset contract

The original Drive archive was supplied and audited on 2026-10-13. Its image contents
remain separate from this code repository. The received archive contains 81 train
and 21 validation images, with one matching label file for every image.
See [`dataset_audit.json`](../experiments/dataset_audit.json) and
[`dataset_manifest.json`](../experiments/dataset_manifest.json) for counts and hashes.
There are 679 training and 159 validation person annotations, and respectively
9 and 1 empty label files. Exact-byte duplicates across splits were not found.

Import the unchanged archive after installing this package:

```bash
python scripts/import_dataset.py /path/to/dataset-20261013T201457Z-1-001.zip --output data/metro
python -m metroyolo.cli audit data/metro
```

The importer retains the original train/valid splits and file bytes, discards caches,
validates data before moving it into place, and creates a local `data.yaml`. It fixes
Colab-specific paths and changes the display name `0` to `person`, keeping class ID 0.
It refuses to overwrite an existing destination. `data/` stays excluded from Git.

The source YAML identifies Roboflow export version 2:
https://universe.roboflow.com/s-workspace-l0vhc/yolometrodataset/dataset/2.
It declares CC BY 4.0; the declaration is recorded metadata, not a verification
of each original image's rights or attribution. The full dataset is not bundled. Three annotated figures from the supplied practice report
are included as historical research examples; see assets/report/README.md.

For another dataset, supply `train/images`, `train/labels`,
`valid/images`, `valid/labels` and a `data.yaml` like [the example](../examples/data.yaml).

## Labels

- One text file per image, sharing its filename stem: `frame_001.jpg` / `frame_001.txt`.
- Each nonempty line: `0 x_center y_center width height` in normalized image coordinates.
- Only class 0 (`person`), finite coordinates, positive width/height and boxes inside the image.
- An **empty text file** explicitly marks a reviewed image containing no people.
- A missing label is an error by default. `--allow-missing-labels` supports legacy datasets
  where absence is known to mean background; do not use it to mask unfinished annotation.
- Two images with the same stem in one split are ambiguous and rejected.
- Images with EXIF orientation must be normalized before annotation so their labels and
  decoded orientation agree. The inference/evaluation loader applies EXIF orientation.

## Audit

```bash
python -m metroyolo.cli audit data/metro --output outputs/data_audit.json
```

The audit reports malformed labels, unreadable images, counts, empty labels, orphan label
files, per-split fingerprints and duplicate image bytes. Exact duplicates across splits are
errors; duplicates within one split are warnings. Different encodings/crops of the same frame
and nearby video frames require scene-level or perceptual review, which is not implemented.

## Preparing splits

Prefer grouping by carriage scene, source video or capture session, with a fixed random seed.
Allocate complete groups to train/validation/test before generating augmentations. Keep a
manifest containing filename, scene/source ID and split. Review validation and test annotations
independently from model predictions. The historical random 81/21 split cannot be reconstructed
from the PDF alone.

## Training

Edit the dataset YAML with an absolute root, audit the data, then use an NVIDIA GPU:

```bash
bash scripts/setup_yolov5.sh
python -m pip install -e '.[training]'
bash scripts/train.sh /absolute/path/to/data.yaml
```

The script uses `--device 0`, 30 epochs, image size 640, batch size 16 and seed 0.
The recovered notebook training log confirms seed 0 and SGD. On Colab put the
repository and data on accessible paths first. Training on a Mac CPU is possible by changing
`--device 0` to `--device cpu`, but the script is set up for GPU training.

YOLOv5 chooses `best.pt` using its detection fitness, not this project's passenger count MAE.
Evaluate `best.pt` independently with the counting CLI. A successful training command does
not establish better counting quality.

## Dataset card to complete before publishing images

Record image sources, rights/permissions, capture conditions, annotation policy, annotation
review, scene grouping, train/validation/test sizes, excluded files and limitations.
No dataset license or permission to redistribute the original photos is inferred from the PDF.
