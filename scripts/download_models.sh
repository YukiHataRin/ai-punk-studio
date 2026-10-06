#!/usr/bin/env bash
# 下載執行所需模型與測試用範例圖
#   models/yolo11n-seg.pt              Ultralytics YOLO11 人體分割（AGPL-3.0）
#   models/pose_landmarker_*.task      MediaPipe Pose Landmarker（Apache-2.0）
#   tests/data/person.jpg              MediaPipe 官方範例圖（僅供測試）
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models tests/data

fetch() {  # fetch <url> <path>
  if [ -f "$2" ]; then echo "已存在：$2"; return; fi
  curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
  echo "已下載：$2"
}

fetch "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt" models/yolo11n-seg.pt
for variant in lite full heavy; do
  fetch "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${variant}/float16/latest/pose_landmarker_${variant}.task" \
        "models/pose_landmarker_${variant}.task"
done
fetch "https://storage.googleapis.com/mediapipe-assets/pose.jpg" tests/data/person.jpg
