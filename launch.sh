#!/usr/bin/env bash
# Linux 啟動 Astra Studio（用專案內的 .conda 環境）。參數會原樣傳下去，例如 ./launch.sh --headless
set -eu
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
exec ./.conda/bin/python -m astra_studio "$@"
