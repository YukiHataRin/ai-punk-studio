# 版本紀錄

版本號採 [語意化版本](https://semver.org/lang/zh-TW/)：`主版號.次版號.修訂號`。
每個版本在 GitHub 上都有對應的 tag（例如 `v0.4.0`），可用 `git checkout v0.4.0` 切換。

## 未發布
- **指標在靜止時不再亂跳**（左右平衡原本會在 0–1 之間跳動）：
  - 只用骨架模型偵測到的關節計算：沒入鏡的腳是模型猜的，抖動原本會被當成動作；看不到雙腳時重心高度與晃動顯示「—」
  - 左右平衡、左右協調、軌跡曲率加上靜止門檻：靜止時平衡接近 1、協調與曲率接近 0，動作明顯時與原公式相同
  - 修正距離跳動：只有臉部少數點量到深度（常是頭部邊緣的背景）時，整副骨架會被推到錯的深度；
    現在推估關節的深度平面只看身體關節，太少時改用整個人的遮罩深度（RTMPose 與 MediaPipe 兩種骨架都適用）
- **專案結構重整**：
  - 新增 `python -m aipunk_studio --export 錄製目錄` 取代 `tools/export_skeleton_csv.py`（匯出移入套件 `io/export.py`）
  - `scripts/download_models.py` → `tools/download_models.py`；`tools/ws_client.py` → `examples/ws_client.py`（與 `web_viewer.html` 同為串流用戶端範例）；`tools/preview.py` → `tools/astra_preview.py`
  - Qt worker 移到 `ui/worker.py`（只有 `ui` 依賴 Qt）；介面啟動流程移到 `apps/gui.py`
  - 模組更名：`perception/pose.py` → `pose_mediapipe.py`、`pose_rtm.py` → `pose_rtmpose.py`、`render/view3d.py` → `cv_view3d.py`
  - 移除舊入口 `main.py`、`scripts/download_models.sh` 與過時的 `docs/PLAN.md`
- WebSocket 串流不再送出骨架模型沒偵測到（3D 也不畫）的關節：`joints` 與 `pixels` 中該位置改為 `null`，陣列長度與關節索引不變；瀏覽器檢視頁同步調整

## 0.6.0 — AI Punk Studio：舞蹈動作指標
- **更名為 AI Punk Studio**（AI Punk 計畫）：GitHub repo 改為 `ai-punk-studio`（舊網址自動轉址）、
  Python 套件改為 `aipunk_studio`、啟動指令改為 `python -m aipunk_studio`
- 整合 [Real-time Dance Aesthetics Analysis](https://github.com/YukiHataRin/realtime-dance-analysis) 的九項動作指標，改為每位舞者各自計算，並使用深度相機的公尺座標
- 指標只使用 13 個關節（鼻、肩、肘、腕、髖、膝、踝）；重心高度改為離地高度（地板由實測腳踝估計）
- 總覽頁新增指標表格；新增「指標」分頁：九項指標即時曲線，左側為每位舞者的照片（代表色外框）
- 錄製、CSV（新增 metrics.csv）、WebSocket 串流與瀏覽器檢視頁都包含指標
- 3D 視圖不再畫出骨架模型沒偵測到的關節
- 修正表格顏色被主題樣式表覆蓋

## 0.5.0 — 跨平台
- 支援 Linux（Ubuntu）與 Windows：攝影機後端依平台（AVFoundation / V4L2 / Media Foundation）
- RTMPose 推論後端 `auto`：macOS 用 CoreML、有 onnxruntime-gpu 時用 CUDA、其餘 CPU
- 模型下載改為跨平台的 `scripts/download_models.py`；新增 `launch.sh`、`launch.bat`
- 介面字型依平台挑選繁體中文字型；新增 `--version`
- GitHub Actions：在 Ubuntu / Windows / macOS 自動跑測試
- 關閉 onnxruntime 內建的 Microsoft 遙測（修正程式結束時偶發的 exit 134 abort）
- 命令列輸出統一為 UTF-8（修正 Windows 非中文主控台印中文時崩潰）

## 0.4.0 — WebSocket 串流與 headless
- WebSocket 即時推送每幀人物資料（ID、距離、3D/2D 關節、輪廓）
- `--headless` 純伺服器模式；擷取迴圈抽成不依賴 Qt 的 `CaptureRunner`
- 範例用戶端 `tools/ws_client.py`、`examples/web_viewer.html`

## 0.3.0 — 攝影機選擇
- 介面可選攝影機；一般攝影機沒有深度時，3D 由身體尺寸估計
- Astra 深度開不起來時自動改用僅 RGB；攝影機沒有影像時報錯而非空等

## 0.2.0 — RTMPose 逐人骨架
- 以 YOLO 人框逐人估計 RTMPose-m（Halpe26，26 點含腳），多人抖動大幅降低
- 用上一幀人框與 YOLO 平行執行；預設模式不再載入 mediapipe

## 0.1.0 — 第一個公開版本
- Astra Pro RGB-D 擷取、軟體深度對齊
- YOLO11 人體分割 + BoT-SORT 追蹤、MediaPipe 骨架、遮罩內取深度提升 3D、One Euro 平滑
- PySide6 介面（五種檢視模式、3D 視圖、人物卡片）、錄製／回放／CSV 匯出
