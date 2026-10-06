"""YOLO11-seg 人體實例分割 + BoT-SORT（ReID）追蹤（移植自 Human Mask Studio 的 studio/engine.py）。

與原版不同：這裡只回傳每個人的遮罩、框與 ID，不在這層合成畫面，
讓融合層可以拿遮罩去取深度。遮罩用 retina_masks 直接輸出原圖解析度，
避免把 letterbox 後的遮罩硬縮放回原圖造成的上下偏移。
"""

import time

import numpy as np

from .. import config as _config  # noqa: F401  確保 YOLO_CONFIG_DIR 先設定好
from ..core.types import SegmentationResult, SegmentedPerson
from .devices import select_device

PERSON_CLASS = 0


class PersonSegmenter:
    def __init__(self, params):
        """params：設定檔的 [segmentation] 區段。"""
        from ultralytics import YOLO

        self.params = params
        self.device = select_device(params.get("device", "auto"))
        self.model = YOLO(params["model"])

    def warmup(self):
        self.model(np.zeros((640, 640, 3), np.uint8), device=self.device.key, classes=[PERSON_CLASS], verbose=False)

    def reset(self):
        """清除追蹤狀態（切換追蹤開關或換來源時呼叫）。"""
        for tracker in getattr(getattr(self.model, "predictor", None), "trackers", None) or []:
            tracker.reset()

    def __call__(self, frame, confidence=None, track=None) -> SegmentationResult:
        p = self.params
        confidence = p["confidence"] if confidence is None else confidence
        track = p["tracking"] if track is None else track
        args = dict(classes=[PERSON_CLASS], conf=confidence, device=self.device.key,
                    imgsz=p["image_size"], retina_masks=True, verbose=False)

        started = time.perf_counter()
        if track:
            result = self.model.track(frame, persist=True, tracker=p["tracker"], **args)[0]
        else:
            result = self.model(frame, **args)[0]
        elapsed = (time.perf_counter() - started) * 1000

        if result.masks is None or result.boxes is None:
            return SegmentationResult([], elapsed)
        masks = result.masks.data.cpu().numpy() > 0.5
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist() if track and result.boxes.id is not None else [None] * len(boxes)
        people = [SegmentedPerson(mask=m, box=b, confidence=float(c), track_id=i)
                  for m, b, c, i in zip(masks, boxes, confs, ids)]
        return SegmentationResult(people, elapsed)
