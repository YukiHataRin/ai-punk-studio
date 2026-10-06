#!/usr/bin/env bash
# 下載執行所需模型與測試用範例圖
#   models/yolo11n-seg.pt              Ultralytics YOLO11 人體分割（AGPL-3.0）
#   models/rtmpose-m_halpe26.onnx      RTMPose-m Halpe26 26 點（OpenMMLab MMPose，Apache-2.0）
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
if [ ! -f models/rtmpose-m_halpe26.onnx ]; then
  zip="models/rtmpose-m_halpe26.zip"
  fetch "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip" "$zip"
  unzip -o -q -j "$zip" "*end2end.onnx" -d models && mv models/end2end.onnx models/rtmpose-m_halpe26.onnx && rm "$zip"
  echo "已解壓：models/rtmpose-m_halpe26.onnx"
else
  echo "已存在：models/rtmpose-m_halpe26.onnx"
fi
fetch "https://storage.googleapis.com/mediapipe-assets/pose.jpg" tests/data/person.jpg
