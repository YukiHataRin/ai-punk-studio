# WebSocket API

AI Punk Studio 用 WebSocket 即時推送每一幀的人物資料：追蹤 ID、距離、3D／2D 關節、人物輪廓與九項舞蹈動作指標。
任何能連 WebSocket 的程式（Python、瀏覽器、TouchDesigner、Unity、Max/MSP…）都可以接收。

- 協定版本：`1`
- 傳輸：WebSocket 文字訊息，內容為 UTF-8 JSON
- 預設位址：`ws://127.0.0.1:8765`

## 開啟串流

```bash
./.conda/bin/python -m aipunk_studio --headless                   # 不開視窗，只當伺服器（預設開啟串流）
./.conda/bin/python -m aipunk_studio --ws                         # 開介面並開啟串流
./.conda/bin/python -m aipunk_studio --headless --ws-host 0.0.0.0 --ws-port 9000   # 開放區域網路、改埠號
```

介面中也可以在右側「串流（WebSocket）」勾選「啟用」，開始或停止擷取時不會中斷連線。

## 連線流程

1. 用戶端連線後，伺服器先送一則 `hello`：影像尺寸、相機內參、骨架格式、指標定義。
2. 之後每處理完一幀就送一則 `frame`。
3. 換攝影機或重新開始擷取時，伺服器會再送一次 `hello`，`frame` 編號從 0 重新計算。用戶端收到 `hello` 時應更新它保存的格式資訊。

```
用戶端                    伺服器
  │──── 連線 ──────────────▶│
  │◀─────────── hello ──────│
  │◀─────────── frame 0 ────│
  │◀─────────── frame 1 ────│
  │──── {"type":"ping"} ───▶│
  │◀──────────── pong ──────│
```

## hello

```jsonc
{
  "type": "hello",
  "protocol": 1,
  "app": "aipunk-studio",
  "coordinates": "RGB 相機座標系：x 右、y 下、z 前，單位公尺；pixels / contour 為 RGB 影像像素",
  "image": {"width": 1280, "height": 720},
  "intrinsics": {"fx": 983.0, "fy": 984.0, "cx": 640.0, "cy": 360.0},
  "has_depth": true,
  "source": {"kind": "astra", "index": 1, "name": "Astra Pro HD Camera"},
  "formats": {
    "halpe26": {
      "names": ["nose", "left_eye", ..., "right_heel"],
      "connections": [[5, 6], [5, 11], ...],
      "hidden": [0, 1, 2, 3, 4, 17, 18, 19]
    },
    "mediapipe33": {"names": [...], "connections": [...], "hidden": [0, 1, ..., 10]}
  },
  "metrics": [
    {"key": "energy", "name": "動作強度", "name_en": "Intensity", "unit": "rad²/s²", "description": "..."},
    ...
  ]
}
```

| 欄位 | 說明 |
|---|---|
| `protocol` | 協定版本，目前為 `1` |
| `image` | RGB 影像尺寸（像素），`pixels`、`box`、`contour` 都在這個座標中 |
| `intrinsics` | RGB 相機內參（像素）。可用來把 3D 關節投影回影像：`u = fx·x/z + cx`、`v = fy·y/z + cy` |
| `has_depth` | 來源是否有深度。`false`（一般攝影機）時所有 3D 座標都是估計值 |
| `source` | 目前的攝影機：`kind` 為 `astra`（RGB + 深度）或 `rgb`（一般攝影機）。回放錄影時可能為 `null` |
| `formats` | 骨架格式：關節名稱（索引即 `joints` 的索引）、要畫的連線、不顯示的關節 |
| `metrics` | 九項指標的定義，順序即建議的顯示順序 |

`formats.*.connections` 只包含兩端都會顯示的連線，可以直接拿來畫骨架。
`hidden` 是不顯示的關節，也就是頭部，跟 Real-time Dance Aesthetics Analysis 一樣只畫肩膀以下。這些關節在 `frame` 中一律是 `null`。

## frame

```jsonc
{
  "type": "frame",
  "frame": 128,
  "t": 4.2701,
  "fps": 30.1,
  "has_depth": true,
  "people": [
    {
      "id": 3,
      "distance": 2.1403,
      "distance_measured": true,
      "centroid": [0.1213, -0.0472, 2.2009],
      "format": "halpe26",
      "joints": [null, null, null, null, null, [0.1832, -0.4121, 2.1544], ...],
      "measured": [0, 0, 0, 0, 0, 1, ...],
      "pixels": [null, null, null, null, null, [724.5, 172.1], ...],
      "visibility": [0.912, 0.887, 0.901, 0.743, 0.802, 0.958, ...],
      "box": [512.3, 88.0, 801.7, 702.4],
      "contour": [[[530, 92], [544, 90], ...]],
      "metrics": {"energy": 1.93021, "sync_velocity": 0.58113, ..., "height": null},
      "metric_joints_measured": 11
    }
  ]
}
```

| 欄位 | 說明 |
|---|---|
| `frame` | 這個連線階段送出的第幾幀，從 0 開始。推送有上限（見 `max_fps`），所以中間不一定每一幀都會送 |
| `t` | 從第一則 `frame` 起算的秒數（單調遞增），計算速度時請用它，不要用收到訊息的時間 |
| `fps` | 伺服器目前的處理速度 |
| `has_depth` | 同 `hello` |
| `people` | 畫面中的每個人，依 `id` 排序；沒有人時為空陣列 |

### people 的欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| `id` | int | BoT-SORT 追蹤 ID。同一個人在畫面中保持同一個 ID；離開後再回來可能是新 ID。數字會持續遞增，不代表人數 |
| `distance` | float \| null | 到相機的距離（公尺）：有骨架時為骨盆中心的距離，否則為遮罩中心的距離 |
| `distance_measured` | bool | `true` 表示距離來自深度實測；`false` 表示整個人都量不到深度，距離由身體尺寸估計（介面顯示為 ≈） |
| `centroid` | [x, y, z] \| null | 人物遮罩深度換算的 3D 位置（公尺）；沒有深度時為 `null` |
| `format` | string | 骨架格式：`halpe26`（RTMPose，預設）或 `mediapipe33`（MediaPipe） |
| `joints` | [[x, y, z] \| null] | 每個關節的 3D 座標（公尺），長度等於該格式的關節數 |
| `measured` | [0 \| 1] | 每個關節是否由深度實測；`0` 為推估 |
| `pixels` | [[u, v] \| null] | 每個關節在 RGB 影像中的像素座標 |
| `visibility` | [float] | 骨架模型對每個關節的信心值（0–1），所有關節都有 |
| `box` | [x1, y1, x2, y2] | 人物框（像素） |
| `contour` | [[[u, v], ...]] | 人物遮罩的外輪廓多邊形（像素，整數），可能有多段，依面積由大到小 |
| `metrics` | object | 九項指標，鍵值見 `hello.metrics`；資料不足時該項為 `null` |
| `metric_joints_measured` | int | 指標用到的 13 個關節中，有幾個由深度實測（0 表示指標完全來自推估骨架） |

**`null` 的關節**：`joints` 與 `pixels` 中為 `null` 的位置，是畫面上不畫的關節，包括：

- 頭部（`hello.formats.*.hidden`）
- 骨架模型判斷沒偵測到的關節（信心值低於 0.5，例如不在畫面內的腳）

陣列長度不變，索引永遠對應 `hello.formats.*.names`，請先檢查 `null` 再使用。

**可能沒有的欄位**：

- 只偵測到人（有遮罩）但沒有骨架時，沒有 `format`、`joints`、`measured`、`pixels`、`visibility`、`metrics`、`metric_joints_measured`。
- 設定關閉輪廓時沒有 `contour`；關閉 2D 資料時沒有 `pixels` 與 `visibility`。

## 座標系與單位

- 3D 座標：**RGB 相機座標系**，原點在 RGB 鏡頭，**x 向右、y 向下、z 朝前**（離開相機），單位公尺。
- 2D 座標：RGB 影像像素，原點在左上角。
- 時間：秒。

## 骨架格式

### halpe26（預設）

| 索引 | 名稱 | 索引 | 名稱 | 索引 | 名稱 |
|---|---|---|---|---|---|
| 0 | nose ✕ | 9 | left_wrist | 18 | neck ✕ |
| 1 | left_eye ✕ | 10 | right_wrist | 19 | hip ✕ |
| 2 | right_eye ✕ | 11 | left_hip | 20 | left_big_toe |
| 3 | left_ear ✕ | 12 | right_hip | 21 | right_big_toe |
| 4 | right_ear ✕ | 13 | left_knee | 22 | left_small_toe |
| 5 | left_shoulder | 14 | right_knee | 23 | right_small_toe |
| 6 | right_shoulder | 15 | left_ankle | 24 | left_heel |
| 7 | left_elbow | 16 | right_ankle | 25 | right_heel |
| 8 | right_elbow | 17 | head ✕ | | |

✕ = 不顯示（`hidden`），一律為 `null`。

### mediapipe33

[MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) 的 33 個關節，索引 0–10（臉部）不顯示。

左右是指被拍攝者自己的左右，不是畫面的左右。

## 指標

| 鍵 | 名稱 | 單位 | 說明 | 何時為 `null` |
|---|---|---|---|---|
| `energy` | 動作強度 | rad²/s² | 肢體角速度平方的加權和 | 沒有任何偵測到的肢體 |
| `sync_velocity` | 左右平衡 | 0–1 | 左右兩側角速度大小的相似度，1 為完全平衡；靜止時接近 1 | 左右兩側沒有都偵測到的肢體 |
| `sync_correlation` | 左右協調 | −1–1 | 左右動作歷史的相關係數，1 為同步、−1 為交替；靜止時接近 0 | 同上 |
| `expansion` | 身體擴展 | m³ | 偵測到的關節構成的凸包體積 | 偵測到的關節少於 4 個 |
| `curvature` | 軌跡曲率 | 1/m | 手腕、腳踝軌跡的平均曲率 | 手腕、腳踝都沒偵測到 |
| `height` | 重心高度 | m | 重心離地高度 | 看不到雙腳，或深度沒量到腳踝（地板未知） |
| `sway` | 重心晃動 | m | 重心偏離兩腳中點的水平距離 | 看不到雙腳 |
| `torque` | 出力程度 | rad/s² | 肢體角加速度的加權和 | 同 `energy` |
| `jerk` | 急動度 | rad²/s⁶ | 角急動度平方的加權和，越低越流暢 | 同 `energy` |

- 每個人剛出現的前 1–2 幀，所有指標都是 `null`：速度需要兩幀，加速度與急動度需要三幀。
- 指標只用 13 個關節（鼻、雙肩、雙肘、雙腕、雙髖、雙膝、雙踝），而且只用骨架模型實際偵測到的。鼻子不顯示，但仍會參與計算。
- `has_depth` 為 `false`，或 `metric_joints_measured` 為 0 時，指標來自推估的 3D 骨架。此時骨架是扁平的，`expansion` 為 0。
- 公式移植自 [Real-time Dance Aesthetics Analysis](https://github.com/YukiHataRin/realtime-dance-analysis)（MIT License），調整處見 `src/aipunk_studio/core/dance_metrics.py` 開頭的說明。

## 用戶端可送的訊息

| 訊息 | 回應 |
|---|---|
| `{"type": "ping"}` | `{"type": "pong", "t": <伺服器 Unix 時間>}` |
| `{"type": "hello"}` | 重送 `hello` |

其他訊息會被忽略，單則訊息上限 64 KB。

## 傳送行為

- **只送最新的一幀**：每個用戶端只保留一則待送的幀。用戶端讀太慢或網路跟不上時，會跳過舊幀直接拿到最新的，不會越積越多、延遲越來越大。所以 `frame` 編號在用戶端看來可能不連續。
- **推送上限**：最多每秒 `max_fps` 幀（預設 30）。
- **沒有用戶端連線時不做序列化**，不影響處理速度。

## 設定

`config/default.toml` 的 `[stream]` 區段：

| 參數 | 預設 | 說明 |
|---|---|---|
| `enabled` | `false` | 介面啟動時是否自動開啟串流（headless 模式一律開啟，除非加 `--no-ws`） |
| `host` | `"127.0.0.1"` | 綁定位址；`"0.0.0.0"` 開放區域網路 |
| `port` | `8765` | 埠號；`0` 表示自動選可用埠 |
| `max_fps` | `30` | 推送上限，`0` 為不限 |
| `contours` | `true` | 是否附上 `contour` |
| `contour_epsilon` | `2.0` | 輪廓簡化程度（像素），越大點越少 |
| `pixels` | `true` | 是否附上 `pixels` 與 `visibility` |

命令列的 `--ws-host`、`--ws-port` 會覆蓋設定檔。

## 範例

### Python

```python
import json
from websockets.sync.client import connect

with connect("ws://127.0.0.1:8765") as ws:
    names = {}
    for raw in ws:
        msg = json.loads(raw)
        if msg["type"] == "hello":
            names = {k: f["names"] for k, f in msg["formats"].items()}
        elif msg["type"] == "frame":
            for p in msg["people"]:
                if "joints" not in p:
                    continue  # 只有遮罩、沒有骨架
                fmt = names[p["format"]]
                wrist = p["joints"][fmt.index("left_wrist")]
                if wrist is not None:
                    print(p["id"], "左手腕", wrist, "強度", p["metrics"]["energy"])
```

更完整的範例見 [`examples/ws_client.py`](../examples/ws_client.py)。

### JavaScript（瀏覽器）

```js
const ws = new WebSocket("ws://127.0.0.1:8765");
let formats = {};
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "hello") formats = msg.formats;
  if (msg.type !== "frame") return;
  for (const p of msg.people) {
    if (!p.pixels) continue;
    for (const [a, b] of formats[p.format].connections) {
      const pa = p.pixels[a], pb = p.pixels[b];
      if (pa && pb) drawLine(pa, pb);  // 沒偵測到的關節為 null，跳過
    }
  }
};
```

會畫出輪廓、骨架、俯視位置圖與指標表格的完整網頁見 [`examples/web_viewer.html`](../examples/web_viewer.html)：在 `examples/` 執行 `python3 -m http.server`，再開 `http://127.0.0.1:8000/web_viewer.html`。

## 安全性

預設只綁 `127.0.0.1`，只有本機能連線。改成 `0.0.0.0` 後，同一網路的任何裝置都能連線讀取資料。資料包含人物位置、骨架與輪廓，連線**沒有加密也沒有驗證**，請只在可信任的網路使用。

## 版本相容

協定版本 `1` 之內只會**新增**欄位，不會改名或刪除既有欄位。用戶端應忽略不認得的欄位。有不相容的變更時，`protocol` 會加 1。
