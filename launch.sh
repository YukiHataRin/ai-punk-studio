#!/usr/bin/env bash
# Linux 啟動 AI Punk Studio（用專案內 uv sync 建立的 .venv）。參數會原樣傳下去，例如 ./launch.sh --headless
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "找不到 .venv，請先執行 uv sync（見 README）" >&2
    exit 1
fi
exec ./.venv/bin/python -m aipunk_studio "$@"
