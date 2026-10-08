"""WebSocket 即時串流：把每幀的人物資料（ID、距離、3D／2D 關節、輪廓）推給所有連線的用戶端。

協定（JSON 文字訊息）：
    連線時伺服器先送 {"type": "hello", ...}：座標系、影像大小、內參、是否有深度、骨架格式（關節名稱與連線）
    之後每幀送 {"type": "frame", "frame": n, "t": 秒, "fps": ..., "has_depth": ..., "people": [...]}
        people 每人欄位與錄製檔 skeleton.jsonl 相同（io/recorder.person_record，含九項指標 "metrics"），另加 "contour"；
        畫面上不畫的關節（頭部，見 hello 的 formats.*.hidden；以及骨架模型判斷沒偵測到的），
        "joints" 與 "pixels" 中該位置為 null（陣列長度不變，索引仍對應關節名稱）
    用戶端可送 {"type": "ping"}，伺服器回 {"type": "pong"}；送 {"type": "hello"} 會重送 hello

每個用戶端只保留一則「待送的最新幀」：上一則還卡在傳送（網路或用戶端讀取跟不上、送出被流量控制擋住）時，
新幀會直接取代還沒送出的舊幀，不會在伺服器端越積越多。
伺服器在獨立執行緒跑 asyncio，publish() 可從任何執行緒呼叫。預設只綁 127.0.0.1。
"""

import asyncio
import json
import threading
import time

import cv2
import numpy as np

from ..core.dance_metrics import METRICS
from ..core.fusion import MIN_VISIBILITY
from ..core.skeleton_format import FORMATS
from .recorder import person_record

PROTOCOL_VERSION = 1


def mask_contours(mask, epsilon=2.0, min_area=200, max_points=400):
    """遮罩 → 外輪廓多邊形（原圖像素座標，整數），以 approxPolyDP 簡化。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            break
        poly = cv2.approxPolyDP(c, epsilon, True)[:, 0, :]
        if len(poly) > max_points:
            poly = poly[np.linspace(0, len(poly) - 1, max_points).astype(int)]
        polys.append(poly.tolist())
    return polys


def hello_message(cfg, has_depth):
    r = cfg["rgb"]
    return {
        "type": "hello", "protocol": PROTOCOL_VERSION, "app": "aipunk-studio",
        "coordinates": "RGB 相機座標系：x 右、y 下、z 前，單位公尺；pixels / contour 為 RGB 影像像素",
        "image": {"width": r["width"], "height": r["height"]},
        "intrinsics": {k: r[k] for k in ("fx", "fy", "cx", "cy")},
        "has_depth": has_depth,
        "source": cfg.get("source"),
        "formats": {name: {"names": list(f.names), "connections": [list(c) for c in f.shown_connections],
                           "hidden": sorted(f.hidden)}
                    for name, f in FORMATS.items()},
        "metrics": [{"key": m.key, "name": m.name, "name_en": m.name_en, "unit": m.unit, "description": m.description}
                    for m in METRICS],
    }


def frame_message(out, metrics, index, t, params):
    people = []
    for p in out.people:
        rec = person_record(p)
        if p.pose is not None:  # 畫面上不畫的關節（頭部、沒偵測到的）不送座標
            fmt = p.pose.fmt
            for j in sorted(set(np.flatnonzero(p.pose.visibility < MIN_VISIBILITY)) | fmt.hidden):
                for key in ("joints", "pixels"):
                    if key in rec:
                        rec[key][j] = None
        if not params.get("pixels", True):
            rec.pop("pixels", None)
            rec.pop("visibility", None)
        if params.get("contours", True) and p.segment is not None:
            rec["contour"] = mask_contours(p.segment.mask, params.get("contour_epsilon", 2.0))
        people.append(rec)
    return {"type": "frame", "frame": index, "t": round(t, 4), "fps": round(metrics.get("fps", 0.0), 1),
            "has_depth": metrics.get("has_depth", True), "people": people}


class _Client:
    def __init__(self):
        self.latest = None
        self.event = asyncio.Event()


class StreamServer:
    def __init__(self, host="127.0.0.1", port=8765, params=None):
        self.host, self.port = host, port
        self.params = params or {}
        self.max_fps = self.params.get("max_fps", 30)
        self._clients = set()
        self._hello = None
        self._loop = None
        self._server = None
        self._thread = None
        self._ready = threading.Event()
        self._error = None
        self._index = 0
        self._t0 = None
        self._last_sent = 0.0

    # ---- 生命週期 ----
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="stream-server")
        self._thread.start()
        self._ready.wait(5)
        if self._error:
            raise RuntimeError(f"WebSocket 伺服器無法啟動（{self.host}:{self.port}）：{self._error}")
        return self

    def _run(self):
        from websockets.asyncio.server import serve

        async def main():
            try:
                self._server = await serve(self._handler, self.host, self.port, max_size=2**16)
            except OSError as error:
                self._error = str(error)
                self._ready.set()
                return
            self.port = self._server.sockets[0].getsockname()[1]  # port=0 時取得實際埠號
            self._ready.set()
            await self._server.wait_closed()

        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    def stop(self):
        if self._loop is not None and self._server is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._server.close)
        if self._thread is not None:
            self._thread.join(timeout=3)

    @property
    def url(self):
        return f"ws://{self.host}:{self.port}"

    @property
    def client_count(self):
        return len(self._clients)

    # ---- 任何執行緒呼叫 ----
    def start_session(self, cfg, has_depth):
        """來源開啟後呼叫：更新 hello 並重送給已連線的用戶端。"""
        self._hello = json.dumps(hello_message(cfg, has_depth), ensure_ascii=False)
        self._index, self._t0 = 0, None
        self._post(self._hello)

    def publish(self, out, metrics):
        now = time.monotonic()
        if not self._clients or (self.max_fps and now - self._last_sent < 1.0 / self.max_fps):
            return
        self._last_sent = now
        if self._t0 is None:
            self._t0 = now
        msg = json.dumps(frame_message(out, metrics, self._index, now - self._t0, self.params), ensure_ascii=False)
        self._index += 1
        self._post(msg)

    def _post(self, msg):
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._deliver, msg)

    # ---- asyncio 執行緒 ----
    def _deliver(self, msg):
        for client in self._clients:
            client.latest = msg  # 只留最新一則，慢的用戶端會跳過舊幀
            client.event.set()

    async def _handler(self, ws):
        client = _Client()
        self._clients.add(client)
        if self._hello is not None:
            await ws.send(self._hello)

        async def sender():
            while True:
                await client.event.wait()
                client.event.clear()
                msg, client.latest = client.latest, None
                if msg is not None:
                    await ws.send(msg)

        task = asyncio.create_task(sender())
        try:
            async for raw in ws:
                try:
                    req = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if req.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong", "t": time.time()}))
                elif req.get("type") == "hello" and self._hello is not None:
                    await ws.send(self._hello)
        except Exception:
            pass  # 用戶端斷線
        finally:
            task.cancel()
            self._clients.discard(client)
