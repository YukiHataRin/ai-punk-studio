"""播放 SessionRecorder 錄下的目錄，介面與 AstraSource 相同（start / stop / latest / error）。

依錄製時的時間戳以實際速度播放；播完後 finished 為 True，latest() 停在最後一幀。
"""

import json
import threading
import time
from pathlib import Path

import cv2

from ..io.recorder import read_skeleton


class PlaybackSource:
    def __init__(self, session_dir, loop=False):
        self.dir = Path(session_dir)
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        if not meta.get("with_images") or not (self.dir / "rgb.mp4").exists():
            raise RuntimeError(f"這個錄製沒有包含影像，無法播放：{self.dir}")
        self.meta = meta
        self.times = [rec["t"] for rec in read_skeleton(self.dir / "skeleton.jsonl")]
        self.loop = loop
        self.error = None
        self.finished = False
        self._lock = threading.Lock()
        self._latest = (None, 0.0, None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="playback")

    def __len__(self):
        return len(self.times)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def latest(self):
        with self._lock:
            return self._latest

    def _read_depth(self, index):
        path = self.dir / "depth" / f"{index:06d}.png"
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None

    def _run(self):
        try:
            while not self._stop.is_set():
                cap = cv2.VideoCapture(str(self.dir / "rgb.mp4"))
                start = time.monotonic()
                for index, t in enumerate(self.times):
                    ok, frame = cap.read()
                    if not ok or self._stop.is_set():
                        break
                    delay = start + t - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                    with self._lock:
                        self._latest = (frame, time.monotonic(), self._read_depth(index))
                cap.release()
                if not self.loop:
                    break
            self.finished = True
        except Exception as error:
            self.error = str(error)
