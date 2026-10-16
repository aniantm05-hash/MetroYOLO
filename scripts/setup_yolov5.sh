#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=915bbf294bb74c859f0b41f1c23bc395014ea679
if [ ! -d vendor/yolov5 ]; then
  mkdir -p vendor
  git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git vendor/yolov5
fi
actual=$(git -C vendor/yolov5 rev-parse HEAD)
if [ "$actual" != "$revision" ]; then
  echo "Unexpected YOLOv5 revision: $actual" >&2
  exit 1
fi
python -m pip install -e '.[inference]'
mkdir -p weights
if [ ! -f weights/yolov5s.pt ]; then
  python - <<'DOWNLOAD'
import os
import urllib.request
from pathlib import Path
url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
path = Path('weights/yolov5s.pt')
temporary = path.with_suffix('.download')
try:
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
DOWNLOAD
fi
python - <<'HASH'
import hashlib
from pathlib import Path
p = Path('weights/yolov5s.pt')
print('YOLOv5 v7.0 ready; weights SHA256:', hashlib.sha256(p.read_bytes()).hexdigest())
HASH
