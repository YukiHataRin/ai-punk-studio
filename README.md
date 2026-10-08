# AI Punk Studio

**同台共舞：多人即時 3D 舞蹈動作解析系統**（AI Punk／AI 叛客 計畫）

用 **Orbbec Astra Pro** 深度相機即時追蹤同台的多位舞者：YOLO11 分割與 BoT-SORT 賦予每人穩定編號，
RTMPose 估計關節，再結合深度換算成真實尺度的 3D 骨架，即時計算動作強度、左右協調、身體擴展、重心穩定、
流暢度等九項指標。附桌面介面、headless 伺服器、WebSocket 串流、錄製與回放。

![演算法架構](docs/figures/architecture_preview.png)

## 專案介紹

- **多人 3D 骨架**：每人 26 個關節（Halpe26，含腳趾、腳跟）的公制 3D 座標（公尺），標明每個關節是深度實測還是推估
- **穩定的人物編號**：YOLO11 像素級遮罩 + BoT-SORT（ReID），人交錯或短暫遮擋仍保留同一 ID；骨架逐人估計，多人時不會互相跳動
- **遮罩內取深度**：關節深度只取自己遮罩內的值，前方的人或背景不會混進來
- **九項舞蹈動作指標**：每位舞者各自計算動作強度、左右平衡與協調、身體擴展、軌跡曲率、重心高度與晃動、出力程度、急動度
- **即時介面**：總覽頁（五種畫面模式、可旋轉 3D 視圖、指標表格）與指標頁（每人照片＋九項指標即時曲線）
- **任何攝影機都能用**：選 Astra Pro 為深度實測，選一般 webcam 時 3D 改由身體尺寸估計
- **WebSocket 串流與 headless 伺服器**：每幀推送 ID、距離、3D／2D 關節、指標與輪廓；可不開視窗只當伺服器（[API 文件](docs/API.md)）
- **錄製與回放**：RGB 影片 + 16-bit 深度 + 每幀 3D 骨架，可不接相機回放，骨架與指標可轉 CSV
- **跨平台**：macOS、Linux、Windows 共用同一份程式

## 安裝

### 需求

| 項目 | 說明 |
|---|---|
| 相機 | Orbbec Astra Pro；沒有時任何 webcam 也能用（3D 為估計）|
| Python | **3.11**（由下方 conda 指令建立）|
| 其他 | conda（Miniconda / Miniforge）、約 4–8 GB 磁碟空間 |

### macOS / Linux

```bash
git clone https://github.com/YukiHataRin/ai-punk-studio.git
cd ai-punk-studio

# 1. 在專案內建立 Python 3.11 環境
conda create -p ./.conda python=3.11 -y

# 2. 安裝套件（要用 NVIDIA GPU 的話，請先看下方「NVIDIA GPU」）
PYTHONNOUSERSITE=1 ./.conda/bin/python -m pip install -r requirements.txt

# 3. ultralytics 會另外裝 opencv-python，與 mediapipe 需要的 contrib 版衝突，只保留 contrib
./.conda/bin/python -m pip uninstall -y opencv-python
./.conda/bin/python -m pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93

# 4. 安裝本專案
./.conda/bin/python -m pip install -e . --no-deps

# 5. 下載模型（YOLO11n-seg、RTMPose-m、MediaPipe Pose）
./.conda/bin/python tools/download_models.py
```

Linux 若出現 `libGLESv2.so.2: cannot open shared object file` 之類的錯誤，請補裝系統函式庫：

```bash
sudo apt install libegl1 libgl1 libgles2 libxkbcommon0 libfontconfig1 libdbus-1-3
```

### Windows（PowerShell）

```powershell
git clone https://github.com/YukiHataRin/ai-punk-studio.git
cd ai-punk-studio
conda create -p .\.conda python=3.11 -y
$env:PYTHONNOUSERSITE = "1"
.\.conda\python.exe -m pip install -r requirements.txt
.\.conda\python.exe -m pip uninstall -y opencv-python
.\.conda\python.exe -m pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93
.\.conda\python.exe -m pip install -e . --no-deps
.\.conda\python.exe tools\download_models.py
```

### NVIDIA GPU（Linux / Windows，選用）

沒有 GPU 也能跑（較慢）。要用 NVIDIA GPU 加速，**請依自己的顯示卡與驅動安裝對應的 CUDA 版 PyTorch**，
並在上面的步驟 2 之前先裝好，`pip install -r requirements.txt` 就會沿用、不會覆蓋：

1. 查驅動支援的 CUDA 版本：執行 `nvidia-smi`，看右上角的 `CUDA Version`（這是驅動**最高**支援的版本）
2. 到 [PyTorch 官網的安裝選擇器](https://pytorch.org/get-started/locally/) 選 Pip、你的系統與**不高於上述版本**的 CUDA，
   複製它給的指令，把開頭的 `pip` 換成專案環境的 Python，例如：

   ```bash
   ./.conda/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

   `cu128` 只是範例，請換成選擇器給你的版本。RTX 50 系列（Blackwell）需要 CUDA 12.8 以上
3. 確認 GPU 可用（應印出 `True` 與顯示卡名稱）：

   ```bash
   ./.conda/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

4. **RTMPose**（選用）：預設在 CPU 上執行即可。要用 GPU 時把 `onnxruntime` 換成
   [`onnxruntime-gpu`](https://onnxruntime.ai/docs/install/)，並依其文件選擇與 CUDA、cuDNN 相容的版本；
   裝好後會自動改用 CUDA，介面右側會顯示「RTMPose-m：CUDA」

### Astra Pro 驅動

- **macOS**：不需要額外驅動
- **Linux**：一般使用者存取 USB 需要 Orbbec 的 udev 規則（需要 sudo 安裝一次），
  見 [pyorbbecsdk 說明](https://github.com/orbbec/pyorbbecsdk)；RGB 鏡頭需要使用者在 `video` 群組
- **Windows**：需安裝 Orbbec 的相機驅動，見 [Orbbec 官網](https://www.orbbec.com/developers/)

### 更新與切換版本

```bash
git pull                                           # 更新到最新版
git checkout v0.6.0                                # 或切到指定版本（版本列表見 Releases）
./.conda/bin/python -m pip install -e . --no-deps  # 切換後重跑，讓版本號更新
```

## 使用方式

### 啟動

```bash
./.conda/bin/python -m aipunk_studio
```

或用啟動檔：macOS 在 Finder 雙擊 `launch.command`、Linux `./launch.sh`、Windows 雙擊 `launch.bat`
（Windows 指令列請把 `./.conda/bin/python` 換成 `.\.conda\python.exe`）。

在右側「來源」選擇攝影機後按「**開始**」：

- 選 **Astra Pro HD Camera（RGB-D）**：深度實測的 3D 骨架。請站在相機前約 **1–3 m**，全身入鏡
- 選**其他攝影機（僅 RGB）**：沒有深度，3D 由身體尺寸估計（距離前標 **≈**）。至少要拍到肩膀或髖部，估計才準
- 插拔攝影機後按「重新整理攝影機」

> **macOS 相機權限**：第一次執行時，請在「終端機」App（或你常用的終端機）中啟動並允許相機存取；
> 也可在「系統設定 → 隱私權與安全性 → 相機」中開啟。MacBook 螢幕蓋上時內建相機無法使用，請改接 USB 攝影機。

常用參數：

```bash
./.conda/bin/python -m aipunk_studio --start --view split   # 開啟即開始，並排顯示 3D
./.conda/bin/python -m aipunk_studio --tab metrics          # 開啟時顯示指標分頁
./.conda/bin/python -m aipunk_studio --camera "j5 WebCam JVCU100"   # 預選攝影機（名稱、裝置 ID 或編號）
./.conda/bin/python -m aipunk_studio --list-cameras         # 列出相機
./.conda/bin/python -m aipunk_studio --play recordings/<時間>   # 回放錄製，不需要相機
./.conda/bin/python -m aipunk_studio --no-seg               # 只跑骨架（不做分割）
./.conda/bin/python -m aipunk_studio --version              # 顯示版本
```

### 介面

| 區域 | 內容 |
|---|---|
| 上方 | RGB / 深度連線燈號、處理 fps |
| 主畫面 | **疊圖**：RGB + 遮罩 + 2D 骨架 + ID／距離<br>**並排 3D**：左疊圖、右 3D 視圖（左鍵拖曳旋轉、滾輪縮放）<br>**深度**：對齊後深度疊在 RGB 上，檢查對齊<br>**僅骨架**：黑底骨架<br>**僅輪廓**：黑底，只畫每人遮罩外框 |
| 指標表格 | 每位舞者一列：ID、距離、實測關節數、九項指標；灰色表示指標來自推估骨架 |
| 指標分頁 | 左側每位舞者的照片（外框為代表色），右側九項指標最近 20 秒的曲線 |
| 右側設定 | 來源、圖層開關、分割信心值、骨架模型、平滑強度、對齊微調、串流（開關與埠號）、錄製 |
| 下方 | 狀態、**● 錄製**、截圖 |

- 實心關節點 = 深度實測，空心 = 推估；沒偵測到的關節不會畫出
- 與原專案相同，骨架只畫肩膀以下，不畫頭部（指標仍以鼻子計算頭部動作）
- ID 只代表「同一個人」，會持續遞增（例如 ID 54），不是人數

> 展出時請避免讓**螢幕、海報上的人像**出現在相機畫面中，系統無法區分它們與真人，也會為它們建立 ID 與指標。

### 舞蹈動作指標

| 指標 | 單位 | 說明 |
|---|---|---|
| 動作強度 | rad²/s² | 肢體角速度平方的加權和，越大動作越激烈 |
| 左右平衡 | 0–1 | 左右兩側角速度大小的相似度，1 為完全平衡（靜止時接近 1） |
| 左右協調 | −1–1 | 左右動作歷史的相關係數，1 為同步、−1 為交替（靜止時接近 0） |
| 身體擴展 | m³ | 關節凸包體積，越大身體越舒展 |
| 軌跡曲率 | 1/m | 手腕、腳踝軌跡的平均曲率，越大動作越圓轉 |
| 重心高度 | m | 重心離地高度（需要深度量到腳踝才有值） |
| 重心晃動 | m | 重心偏離兩腳中點的水平距離（需要看到雙腳） |
| 出力程度 | rad/s² | 肢體角加速度的加權和 |
| 急動度 | rad²/s⁶ | 角急動度平方的加權和，越低越流暢 |

指標只用 13 個關節（鼻、雙肩、雙肘、雙腕、雙髖、雙膝、雙踝），而且只用骨架模型實際偵測到的：例如腳不在畫面內時，強度等指標只算上半身，重心高度與晃動顯示「—」。指標是動作的描述值；人站得越遠、全身入鏡、深度量得越完整，數值越穩定。

### 錄製與匯出

按「● 錄製」開始、再按一次停止，存到 `recordings/<時間>/`：

| 檔案 | 內容 |
|---|---|
| `rgb.mp4` | 原始 RGB，可用 `--play` 重新跑整條流程 |
| `depth/*.png` | 16-bit 深度（mm，0 = 無效）|
| `skeleton.jsonl` | 每幀每人的 ID、距離、3D 關節、指標 |
| `meta.json` | 相機內外參 |

右側「錄製」區取消「包含 RGB-D 影像」時只寫 `skeleton.jsonl`（檔案小很多，但無法回放）。

轉成 CSV（輸出關節座標與 `metrics.csv`，可直接用 Excel / pandas 開）：

```bash
./.conda/bin/python -m aipunk_studio --export recordings/<時間>
```

座標系為 RGB 相機座標系：**x 向右、y 向下、z 朝前**，單位公尺。

### WebSocket 串流與 Headless 伺服器

不開視窗、只當伺服器：

```bash
./.conda/bin/python -m aipunk_studio --headless                          # 自動選 Astra Pro，ws://127.0.0.1:8765
./.conda/bin/python -m aipunk_studio --headless --camera "j5 WebCam JVCU100"
./.conda/bin/python -m aipunk_studio --headless --ws-host 0.0.0.0        # 開放區域網路其他裝置連線
./.conda/bin/python -m aipunk_studio --headless --record                 # 同時錄製
./.conda/bin/python -m aipunk_studio --headless --play recordings/<時間>   # 用錄影當來源
```

`Ctrl-C` 結束（或加 `--duration 秒數`）。有視窗時，在右側「串流（WebSocket）」勾選「啟用」，或啟動時加 `--ws`；埠號可在同一區修改（串流關閉時才能改，0 為自動選可用埠），也可以用 `--ws-port` 指定。

用戶端範例：

```bash
./.conda/bin/python examples/ws_client.py                  # 終端機印出每幀摘要
./.conda/bin/python examples/ws_client.py --json > s.jsonl # 存原始訊息
```

瀏覽器：在 `examples/` 執行 `python3 -m http.server`，再開 `http://127.0.0.1:8000/web_viewer.html`，會即時畫出輪廓、骨架、位置與指標。

訊息格式、每個欄位的意義、座標系、`null` 的規則與 Python / JavaScript 範例，請參閱 **[WebSocket API 文件](docs/API.md)**。

> **安全性**：預設只綁 `127.0.0.1`。改成 `0.0.0.0` 後同一網路的任何裝置都能連線讀取資料，連線**沒有加密也沒有驗證**，請只在可信任的網路使用。

### 設定

所有參數在 [`config/default.toml`](config/default.toml)，常調的有：

| 區段 | 參數 | 說明 |
|---|---|---|
| `[stream]` | `host` `port` `contours` `max_fps` | WebSocket 綁定位址、埠號、是否附輪廓、推送上限 |
| `[webcam]` | `hfov_deg` | 一般攝影機的水平視角（預設 70°，依攝影機規格調整可讓估計距離更準）|
| `[skeleton]` | `est_torso_m` `est_shoulder_m` | 沒有深度時估計距離用的身體尺寸 |
| `[extrinsics]` | `translation` | 深度與 RGB 的對齊；可在介面「深度」模式用滑桿微調後填回 |
| `[segmentation]` | `confidence` `tracking` | 分割信心值、是否啟用 BoT-SORT |
| `[pose]` | `backend` `num_poses` | `rtmpose`（預設）或 `mediapipe`；最多估計骨架的人數 |
| `[smoothing]` | `min_cutoff_3d` `beta_3d` | 抖動就調小 `min_cutoff`，動作太黏就調大 `beta` |

> 選用 `mediapipe` 骨架時，MediaPipe 會連線 Google 回傳使用資料（官方未提供關閉方式）；預設的 `rtmpose` 不會。

## 授權

[AGPL-3.0](LICENSE)。舞蹈動作指標移植自 [Real-time Dance Aesthetics Analysis](https://github.com/YukiHataRin/realtime-dance-analysis)（MIT License），第三方授權全文見 `licenses/`。
