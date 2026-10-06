# Astra Studio 整合規劃

把兩個專案合成一個桌面 App：

- **orbbec-astra-pro**（本專案）：Astra Pro RGB-D 擷取、軟體 D2C 對齊、MediaPipe 多人骨架、深度提升成 3D、One Euro 平滑
- **[human_semantic_segmentation](https://github.com/YukiHataRin/human_semantic_segmentation)**（Human Mask Studio）：PySide6 + qt-material 原生 UI、YOLO11n-seg 人體遮罩、BoT-SORT ReID 追蹤 ID、相機列舉、背景 worker

## 1. 為什麼要合：兩邊互補

| 現在的弱點（本專案） | Human Mask Studio 補上的能力 | 合併後 |
|---|---|---|
| 深度取樣靠「軀幹深度 ±0.6 m」猜哪些是人 | 每個人的像素級遮罩 | **遮罩內取深度**：只採該人遮罩（內縮幾 px）裡的深度，背景與別人都不會混進來 |
| 自寫貪婪配對，人交錯就可能換號 | BoT-SORT + ReID，ID 穩定 | 以 YOLO 的 track ID 當人物 ID，骨架掛在 ID 底下 |
| MediaPipe 多人模式在人重疊時不穩 | 每人一個 bbox | 第二階段改成「每人裁切後跑單人 pose」 |
| OpenCV 視窗、按鍵操作 | 完整 Qt UI、設定即時生效、錯誤提示 | 統一用 Qt UI |

## 2. 每幀資料流

```
Astra Pro ─┬─ RGB 1280×720 ──┬─► YOLO11n-seg + BoT-SORT（MPS）─► 遮罩、bbox、track ID ─┐
           │                 └─► MediaPipe Pose（CPU）─────────► 33 關節 × N 人 ──────┤
           └─ 深度 640×480 ──► D2C 對齊（RGB 視角 640×360）───────────────────────────────┤
                                                                                         ▼
                      融合：pose ↔ track 配對（關節落在遮罩內的比例，Hungarian）
                                                                                         ▼
                      提升：遮罩內深度中位數 → 公制 3D；量不到的用 world landmarks 補
                                                                                         ▼
                      One Euro（以 track ID 為 key）→ 2D 疊圖 / 3D 視圖 / 錄製 / 匯出
```

YOLO 與 MediaPipe 互不相依，可以兩條執行緒平行跑：YOLO 用 MPS、MediaPipe 用 CPU（GPU delegate 在 macOS 會洩漏記憶體），實測每幀約 27 ms。

## 3. 專案結構（建議）

```
orbbec-astra-pro/                 # 整合後的主專案（human_semantic_segmentation 保持不動）
├── pyproject.toml                # 套件設定 + 入口指令 astra-studio
├── requirements.txt
├── launch.command                # 雙擊啟動（相機權限歸終端機）
├── config/
│   └── default.toml              # 相機內外參、模型、門檻、平滑、UI 預設
├── models/                       # yolo11n-seg.pt、pose_landmarker_*.task（不進版控）
├── src/astra_studio/
│   ├── __main__.py               # CLI 參數、建立 Qt App、套主題
│   ├── config.py
│   ├── core/                     # 純演算法，不依賴 Qt，可單元測試
│   │   ├── types.py              # RGBDFrame、PersonObs、Skeleton3D、FrameResult 等 dataclass
│   │   ├── geometry.py           # 內參、反投影、旋轉
│   │   ├── registration.py       # D2C（← skeleton3d/registration.py）
│   │   ├── fusion.py             # pose ↔ track 配對【新】
│   │   ├── lift.py               # 遮罩取樣 + 軀幹門檻備援（← skeleton3d/lift.py）
│   │   └── filters.py            # One Euro（← skeleton3d/filters.py）
│   ├── sensors/
│   │   ├── base.py               # FrameSource 介面 + LatestFrame 信箱（← studio/workers.py）
│   │   ├── astra.py              # RGB + 深度配對輸出（← skeleton3d/camera.py）
│   │   ├── webcam.py             # 一般相機：沒有深度，只做 2D
│   │   ├── playback.py           # 播放錄製檔
│   │   └── discovery.py          # 相機列舉（← studio/cameras.py）
│   ├── perception/
│   │   ├── devices.py            # CUDA / MPS / CPU 選擇（← studio/engine.py）
│   │   ├── segmentation.py       # YOLO11-seg + BoT-SORT（← studio/engine.py）
│   │   └── pose.py               # MediaPipe（← skeleton3d/pose.py）
│   ├── pipeline/
│   │   ├── pipeline.py           # 單幀流程：感測 → 分割/骨架 → 融合 → 提升 → 平滑
│   │   └── worker.py             # QThread、設定信箱、效能統計（← studio/workers.py）
│   ├── render/
│   │   ├── overlay.py            # 遮罩 + 骨架 + ID 標籤（合併 compose_masks 與 draw_2d）
│   │   └── view3d.py             # 3D 骨架視圖（← skeleton3d/viz.py）
│   ├── io/
│   │   ├── recorder.py           # 錄 RGB-D + 結果
│   │   └── export.py             # 3D 骨架 JSON / CSV、截圖
│   └── ui/
│       ├── theme.py              # dark_teal + 自訂樣式（← studio/__main__.py）
│       ├── main_window.py
│       └── widgets/
│           ├── viewport.py       # 主畫面（疊圖 / 並排 3D / 深度 / 僅骨架）
│           ├── view3d_widget.py  # 可拖曳旋轉的 3D 視圖
│           ├── people_panel.py   # 每人卡片：ID、距離、實測關節數
│           ├── inspector.py      # 右側設定面板
│           └── status_bar.py
├── tools/
│   ├── preview.py
│   ├── calibrate_rgbd.py         # 棋盤格 RGB-D 校正，結果寫回 config【新】
│   └── make_architecture_figure.py
├── tests/
│   ├── test_registration.py  test_lift.py  test_fusion.py  test_filters.py
│   ├── test_pipeline.py          # 用假感測器跑完整流程
│   └── test_ui.py                # 視窗開關、設定更新（沿用 test_native.py 寫法）
├── docs/
│   ├── PLAN.md  figures/
├── recordings/  captures/        # 輸出（不進版控）
└── skeleton3d/ + main.py         # 現有版本，遷移完成後刪除
```

分層規則：`core` 不 import Qt、torch、mediapipe；`perception` 不 import Qt；只有 `ui` 和 `pipeline/worker.py` 碰 Qt。這樣演算法都能不開相機測試。

## 4. UI 設計

沿用 Human Mask Studio 的版面與 dark_teal 主題，讓兩個 App 看起來是同一家：

- **上方狀態列**：RGB / 深度連線燈號、運算裝置、fps
- **主畫面**：四種模式切換
  - 疊圖：RGB + 遮罩 + 2D 骨架 + ID/距離標籤
  - 並排 3D：左疊圖、右 3D 視圖
  - 深度：對齊後深度 + 骨架，用來檢查對齊
  - 僅骨架：黑底
- **人物列**：每個 ID 一張卡，顯示顏色、距離、實測關節比例（越低代表越多推估）
- **右側設定面板**：來源、圖層開關、分割（信心值、BoT-SORT）、骨架（模型、平滑強度）、深度（遮罩取樣、對齊微調）、各階段耗時
- **下方操作列**：停止、錄製、截圖、匯出 3D
- 遮罩、骨架、ID 標籤、卡片都用 **同一個 track 顏色**，一眼就能對上是誰

## 5. 環境

- 統一用本專案的 `.conda`（**Python 3.11**）。pyorbbecsdk 1.3.2 在 macOS 只有 cp311 的 wheel，所以不能用 Human Mask Studio 的 3.12。
- 要加裝：`torch`、`torchvision`、`ultralytics`、`PySide6`、`qt-material`、`cv2-enumerate-cameras`、`lap`、`pyqtgraph`、`PyOpenGL`
- OpenCV 衝突：ultralytics 依賴 `opencv-python`，mediapipe 依賴 `opencv-contrib-python`。裝完後只保留 contrib（功能是 opencv-python 的超集）。

## 6. 實作階段

1. **重構與合併環境**：建立 `src/astra_studio` 骨架，把 skeleton3d 搬進對應模組，移植 YOLO 引擎；現有功能（OpenCV 視窗版）照常能跑
2. **融合**：YOLO 遮罩取深度、BoT-SORT ID 取代自寫追蹤、pose ↔ track 配對；單元測試
3. **Qt UI**：主視窗、四種檢視、人物列、設定面板、即時設定更新
4. **錄製 / 匯出 / 校正**：錄 RGB-D、匯出 3D 骨架、棋盤格校正工具
5. （選做）每人裁切跑單人 pose，改善人重疊時的骨架品質

## 7. 已決定

- 整合位置：**orbbec-astra-pro**；human_semantic_segmentation 不動，所需模組複製過來
- 3D 視圖：**pyqtgraph OpenGL**（`GLViewWidget`），畫骨架、地板網格，並可畫每人的遮罩深度點雲
  - `render/view3d.py` 改成產生 pyqtgraph 的 GL 物件資料；`ui/widgets/view3d_widget.py` 負責顯示與滑鼠互動
  - 需加裝 `pyqtgraph`、`PyOpenGL`
- UI 語言：**中文**
