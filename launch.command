#!/bin/zsh
# 雙擊啟動 AI Punk Studio（用專案內 uv sync 建立的 .venv）。相機權限屬於啟動它的終端機 App。
set -eu
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
    echo "找不到 .venv，請先執行 uv sync（見 README）"
    read -k 1 "?按任意鍵關閉"
    exit 1
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec ./.venv/bin/python -m aipunk_studio "$@"
