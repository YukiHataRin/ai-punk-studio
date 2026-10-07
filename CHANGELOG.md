# 版本紀錄

版本號採 [語意化版本](https://semver.org/lang/zh-TW/)：`主版號.次版號.修訂號`。
每個版本在 GitHub 上都有對應的 tag（例如 `v0.4.0`），可用 `git checkout v0.4.0` 切換。

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
