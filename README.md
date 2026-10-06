# Astra Multi-Person Skeleton

用 **Orbbec Astra Pro** RGB-D 相機做**多人 3D 骨架追蹤**：
YOLO11 人體分割 + BoT-SORT 追蹤 ID + MediaPipe Pose，再以深度相機把每個關節換算成公制 3D 座標，
附原生桌面介面（Astra Studio）、錄製與回放。

![演算法架構](docs/figures/architecture_preview.png)

## 功能

- **多人 3D 骨架**：每人 33 個關節的公制 3D 座標（公尺），標明每個關節是深度實測還是推估
- **人體分割與穩定 ID**：YOLO11 像素級遮罩 + BoT-SORT（ReID），人交錯或短暫遮擋仍保留同一 ID
- **遮罩內取深度**：關節深度只取自己遮罩內的值，前方的人或背景不會混進來；被擋住的關節改用推估
- **即時介面**：疊圖／並排 3D／深度／僅骨架／僅輪廓五種模式、可旋轉 3D 視圖（含點雲）、每人卡片、即時調參
- **錄製與回放**：RGB 影片 + 16-bit 深度 + 每幀 3D 骨架，可不接相機回放，骨架可轉 CSV
- **約 30 fps**：分割（GPU）與骨架（CPU）平行推論，每幀約 28 ms（Apple M5、1280×720）

## 需求

| 項目 | 說明 |
|---|---|
| 相機 | Orbbec Astra Pro（深度 PID 0x0403 + RGB「Astra Pro HD Camera」）|
| 系統 | macOS（Apple Silicon 實測）；Linux / Windows 理論上可行但未測試 |
| Python | **3.11**（pyorbbecsdk 1.3.2 在 macOS 只有 cp311 wheel）|
| 其他 | conda（Miniconda / Miniforge）、約 2 GB 磁碟空間 |

## 安裝

```bash
git clone https://github.com/YukiHataRin/astra-multi-person-skeleton.git
cd astra-multi-person-skeleton

# 1. 在專案內建立 Python 3.11 環境
conda create -p ./.conda python=3.11 -y

# 2. 安裝套件
PYTHONNOUSERSITE=1 ./.conda/bin/python -m pip install -r requirements.txt

# 3. ultralytics 會另外裝 opencv-python，與 mediapipe 需要的 contrib 版衝突，只保留 contrib
./.conda/bin/python -m pip uninstall -y opencv-python
./.conda/bin/python -m pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93

# 4. 安裝本專案（可編輯模式）
./.conda/bin/python -m pip install -e . --no-deps

# 5. 下載模型（YOLO11n-seg、MediaPipe Pose lite/full/heavy）與測試圖
./scripts/download_models.sh
```

## 啟動

接上 Astra Pro 後：

```bash
./.conda/bin/python -m astra_studio
```

或在 Finder 雙擊 `launch.command`。視窗開啟後按右側「**開始**」。

> **macOS 相機權限**：權限屬於「啟動程式的那個 App」。第一次執行時，請在「終端機」App
> （或你常用的終端機）中啟動並允許相機存取；也可在「系統設定 → 隱私權與安全性 → 相機」中開啟。
> 從背景程序或沒有權限的 App 啟動時，深度會正常但 RGB 打不開。

常用參數：

```bash
./.conda/bin/python -m astra_studio --start --view split   # 開啟即開始，並排顯示 3D
./.conda/bin/python -m astra_studio --no-seg               # 只跑骨架（不做分割）
./.conda/bin/python -m astra_studio --play recordings/20261006_112336   # 回放錄製，不需要相機
./.conda/bin/python -m astra_studio --list-cameras         # 列出相機
./.conda/bin/python -m astra_studio --cv                   # OpenCV 簡易檢視器（開發用）
```

## 介面

| 區域 | 內容 |
|---|---|
| 上方 | RGB / 深度連線燈號、處理 fps |
| 主畫面 | **疊圖**：RGB + 遮罩 + 2D 骨架 + ID／距離<br>**並排 3D**：左疊圖、右 3D 視圖（左鍵拖曳旋轉、滾輪縮放）<br>**深度**：對齊後深度疊在 RGB 上，檢查對齊<br>**僅骨架**：黑底骨架<br>**僅輪廓**：黑底，只畫每人遮罩外框 |
| 人物卡片 | 每個 ID 一張：距離、實測關節數（顏色與畫面中一致）|
| 右側設定 | 圖層開關、分割信心值、BoT-SORT 開關、骨架模型（lite/full/heavy）、平滑強度、遮罩取樣、對齊微調、各階段耗時 |
| 下方 | 狀態、**● 錄製**、截圖 |

- 實心關節點 = 深度實測，空心 = 推估
- 距離前有 **≈**（OpenCV 畫面為 `~`）表示這個人完全沒量到深度（太近 < 0.4 m、太遠或被遮擋），數值為推估
- ID 由 BoT-SORT 產生並持續遞增（例如 ID 54），只代表「同一個人」，不是人數

## 錄製與匯出

按「● 錄製」開始、再按一次停止，存到 `recordings/<時間>/`：

| 檔案 | 內容 |
|---|---|
| `rgb.mp4` | 原始 RGB（不含疊圖），可用 `--play` 重新跑整條管線 |
| `depth/*.png` | 16-bit 深度（mm，0 = 無效），與影片逐幀對應 |
| `skeleton.jsonl` | 每幀一行：每人 ID、距離（是否實測）、33 關節 3D 座標與是否實測、遮罩框 |
| `meta.json` | 錄製當下的相機內外參 |

右側「錄製」區取消「包含 RGB-D 影像」時只寫 `skeleton.jsonl`（檔案小很多，但無法回放）。

轉成 CSV（每列一個關節，可直接用 Excel / pandas 開）：

```bash
./.conda/bin/python tools/export_skeleton_csv.py recordings/<時間>
```

座標系：RGB 相機座標系，**x 向右、y 向下、z 朝前（離開相機）**，單位公尺。

## 設定

所有參數在 [`config/default.toml`](config/default.toml)，常調的有：

| 區段 | 參數 | 說明 |
|---|---|---|
| `[rgb]` `[depth]` | `fx` `fy` `cx` `cy` | 相機內參（目前依原廠 FOV 推算）|
| `[extrinsics]` | `translation` | 深度 → RGB 相機的平移；可在介面「深度」模式下用滑桿微調後填回 |
| `[segmentation]` | `confidence` `tracking` | YOLO 信心值、是否啟用 BoT-SORT |
| `[pose]` | `model` `num_poses` | 骨架模型（heavy 最穩）、最多人數 |
| `[fusion]` | `use_mask` `point_cloud` | 遮罩內取深度、3D 點雲 |
| `[skeleton]` | `depth_gate` | 關節深度與人體基準深度差超過此值視為錯值 |
| `[smoothing]` | `min_cutoff_3d` `beta_3d` | One Euro 平滑：抖動就調小 `min_cutoff`，太黏就調大 `beta` |

## 演算法

每幀流程（詳見上方架構圖）：

1. **擷取**：RGB（UVC，1280×720）與深度（OpenNI，640×480）各一條執行緒，只取最新影格
2. **分割 ∥ 骨架**：YOLO11n-seg + BoT-SORT（GPU / MPS）與 MediaPipe Pose（CPU）在兩條執行緒平行執行
3. **深度對齊**：以內外參把深度圖投影到 RGB 視角（軟體 D2C）
4. **融合**：依「骨架可見關節落在遮罩內的比例」把骨架配對到遮罩；關節深度只取自己遮罩內、且與人體基準深度差 < 0.6 m 的值，取中位數 + 5 cm（表面 → 關節中心）
5. **補值**：量不到深度的關節，以 MediaPipe world landmarks 骨架平移到實測關節上補齊；只有遮罩沒有骨架的人，以遮罩深度中位數給出 3D 位置
6. **平滑**：以 track ID 為 key 的 One Euro Filter

## 專案結構

```
├── config/            default.toml（所有參數）、botsort_reid.yaml
├── src/astra_studio/
│   ├── core/          純演算法（不依賴 Qt / torch / mediapipe）：對齊、融合、3D 提升、平滑
│   ├── sensors/       Astra Pro 擷取、錄製回放、相機列舉
│   ├── perception/    YOLO11 分割 + BoT-SORT、MediaPipe Pose、推論裝置選擇
│   ├── pipeline/      單幀流程（分割 ∥ 骨架 ∥ 對齊 → 融合 → 平滑）、Qt 背景 worker
│   ├── render/        2D 疊圖、3D 場景資料、ID 色票
│   ├── io/            錄製
│   ├── ui/            PySide6 介面（主視窗、3D 視圖、人物卡片、設定面板）
│   └── apps/          OpenCV 簡易檢視器
├── tests/             core / perception / ui / io 測試
├── tools/             export_skeleton_csv.py、preview.py（原始畫面）、make_architecture_figure.py
├── scripts/           download_models.sh
└── docs/              PLAN.md（整合規劃）、figures/（架構圖）
```

## 測試

```bash
PYTHONNOUSERSITE=1 ./.conda/bin/python -m unittest discover -s tests -v
```

不需要相機。`test_core` / `test_ui` / `test_io` 用合成資料；`test_perception` 用範例圖跑真實模型
（需先執行 `scripts/download_models.sh`，否則自動略過），並包含 GPU 記憶體不成長的回歸測試。

## 已知限制

- **對齊是估計值**：Astra Pro 的硬體深度對齊在 macOS 上會卡住，SDK 也讀不到出廠校正，內外參依原廠 FOV 推算，3D 位置可能有數公分誤差。可在「深度」模式用「對齊 x 平移」微調。
- **深度有效範圍約 0.4–6 m**：太近或太遠時整個人改用推估（距離前標 ≈）。要量到全身，請站在相機前約 1–3 m。
- **MediaPipe 固定用 CPU**：macOS 上 mediapipe 的 GPU delegate 每幀洩漏約 14 MB Metal 記憶體（0.10.35 與 1.0.1 皆然），30 fps 下不到一分鐘耗盡；1.0.1 的 CPU delegate 又會崩潰，因此固定 0.10.35 + CPU。
- **MediaPipe 0.10.35 的使用資料回傳**：此版本會嘗試連線 `play.googleapis.com`（log 中可見 `portable_clearcut_uploader`），官方未提供關閉方式（[google-ai-edge/mediapipe#6291](https://github.com/google-ai-edge/mediapipe/issues/6291)）。介意者可用防火牆（如 LuLu、Little Snitch）阻擋。
- **Astra Pro 的 RGB 與深度不同步**：兩者是獨立裝置，快速動作時可能有一兩幀時間差。

## 致謝

- [Orbbec pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)：Astra Pro 深度擷取（v1，支援 OpenNI 協定裝置）
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)：Pose Landmarker
- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLO11 實例分割與追蹤整合
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)：多目標追蹤與 ReID
- [PySide6](https://doc.qt.io/qtforpython-6/)、[qt-material](https://github.com/UN-GCPDS/qt-material)、[pyqtgraph](https://github.com/pyqtgraph/pyqtgraph)：介面與 3D 視圖
- [Human Mask Studio](https://github.com/YukiHataRin/human_semantic_segmentation)：分割引擎、相機列舉與介面風格的來源
- One Euro Filter：Casiez, Roussel & Vogel, *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems*, CHI 2012

## 授權

[AGPL-3.0](LICENSE)。本專案使用的 Ultralytics YOLO 以 AGPL-3.0 授權；MediaPipe、pyorbbecsdk 為 Apache-2.0。
