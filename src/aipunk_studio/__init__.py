"""AI Punk Studio：同台共舞——以 Orbbec Astra Pro 深度相機做多人即時 3D 舞蹈動作解析。"""

__version__ = "0.6.0"

import os
import sys

# 關閉第三方函式庫的遙測，必須在它們初始化之前設定（套件匯入時最先執行）：
# onnxruntime 1.21+ 內建 Microsoft 遙測（送往 mobile.events.data.microsoft.com），
# 它的上傳執行緒在程式結束時會鎖到已銷毀的 mutex 而 abort（exit 134）。
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

def utf8_stdio():
    """Windows 主控台常是 cp1252 等編碼，印中文會 UnicodeEncodeError；統一改成 UTF-8，無法顯示的字元以 ? 取代。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
