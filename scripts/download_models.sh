#!/usr/bin/env bash
# 相容舊指令：實際下載由跨平台的 download_models.py 處理
set -euo pipefail
cd "$(dirname "$0")/.."
PY="./.conda/bin/python"; [ -x "$PY" ] || PY="python3"
exec "$PY" scripts/download_models.py "$@"
