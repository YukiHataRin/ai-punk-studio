#!/bin/zsh
# 雙擊啟動 Astra Studio（用專案內的 .conda 環境）。相機權限屬於啟動它的終端機 App。
set -eu
cd "${0:A:h}"
export PYTHONNOUSERSITE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec ./.conda/bin/python -m astra_studio "$@"
