"""錄製一次擷取：原始 RGB 影片、16-bit 深度序列、每幀 3D 骨架。

目錄格式（recordings/<YYYYmmdd_HHMMSS>/）：
    meta.json        相機內外參、設定、開始時間
    rgb.mp4          原始 RGB（不含疊圖），可重新跑整條管線
    depth/000000.png 16-bit 深度（mm，0 = 無效），與 rgb.mp4 的幀一一對應；沒有深度的幀不寫檔
    skeleton.jsonl   每幀一行：時間、每人 ID / 距離 / 33 關節 3D 座標與是否實測

寫檔在背景執行緒進行，不拖慢管線。
"""

import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import PROJECT_ROOT

FORMAT_VERSION = 1


def _r(a, nd=4):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def person_record(p):
    rec = {"id": p.track_id, "distance": None if p.distance is None else round(p.distance, 4),
           "distance_measured": p.distance_measured,
           "centroid": None if p.centroid is None else _r(p.centroid)}
    if p.skeleton is not None:
        rec["joints"] = _r(p.skeleton.points)
        rec["measured"] = p.skeleton.measured.astype(int).tolist()
    if p.pose is not None:
        rec["pixels"] = _r(p.pose.pixels, 1)
        rec["visibility"] = _r(p.pose.visibility, 3)
    if p.segment is not None:
        rec["box"] = _r(p.segment.box, 1)
    return rec


class SessionRecorder:
    def __init__(self, cfg, root=None, with_images=True, fps=30.0):
        self.dir = Path(root or PROJECT_ROOT / "recordings") / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=False)
        self.with_images = with_images
        self.fps = fps
        self.index = 0
        self.t0 = None
        self._video = None
        self._jsonl = open(self.dir / "skeleton.jsonl", "w", encoding="utf-8")
        if with_images:
            (self.dir / "depth").mkdir()
        meta = {"format": FORMAT_VERSION, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "fps": fps,
                "with_images": with_images,
                "rgb": cfg["rgb"], "depth": cfg["depth"], "extrinsics": cfg["extrinsics"],
                "coordinates": "RGB 相機座標系：x 右、y 下、z 前，公尺"}
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self._queue = queue.Queue(maxsize=120)
        self._error = None
        self._thread = threading.Thread(target=self._writer, daemon=True, name="recorder")
        self._thread.start()

    def write(self, frame, depth_mm, output, stamp):
        """在管線執行緒呼叫；只把資料丟進佇列。"""
        if self._error:
            raise RuntimeError(f"錄製失敗：{self._error}")
        if self.t0 is None:
            self.t0 = stamp
        line = json.dumps({"frame": self.index, "t": round(stamp - self.t0, 4),
                           "people": [person_record(p) for p in output.people]}, ensure_ascii=False)
        self._queue.put((self.index, frame if self.with_images else None,
                         depth_mm if self.with_images else None, line))
        self.index += 1

    def _writer(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                index, frame, depth, line = item
                self._jsonl.write(line + "\n")
                if frame is not None:
                    if self._video is None:
                        h, w = frame.shape[:2]
                        self._video = cv2.VideoWriter(str(self.dir / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                    self._video.write(frame)
                if depth is not None:
                    cv2.imwrite(str(self.dir / "depth" / f"{index:06d}.png"), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        except Exception as error:
            self._error = str(error)

    def close(self):
        """等待佇列寫完並關檔，回傳錄製目錄。"""
        self._queue.put(None)
        self._thread.join()
        self._jsonl.close()
        if self._video is not None:
            self._video.release()
        if self._error:
            raise RuntimeError(f"錄製失敗：{self._error}")
        return self.dir


def read_skeleton(path):
    """逐行讀取 skeleton.jsonl。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
