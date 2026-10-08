"""AI Punk Studio：同台共舞——以 Orbbec Astra Pro 深度相機做多人即時 3D 舞蹈動作解析。

分層（上層可以 import 下層，下層不 import 上層）：

    ui · apps                              Qt 介面 · headless 伺服器與其他入口
    pipeline                               單幀流程與擷取迴圈（介面與 headless 共用）
    sensors · perception · io · render     影像來源 · 模型推論 · 錄製與串流 · 繪圖資料
    core                                   純演算法（numpy、scipy、OpenCV）

只有 ui 依賴 Qt（PySide6、pyqtgraph），其餘各層都能在沒有介面的環境中執行與測試。
"""

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
