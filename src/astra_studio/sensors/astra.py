"""Orbbec Astra Pro：RGB 走 UVC（OpenCV / AVFoundation），深度走 pyorbbecsdk（OpenNI 協定）。"""

import cv2
import numpy as np

from .base import Grabber


class RgbCamera(Grabber):
    name = "astra-rgb"

    def __init__(self, index, width, height):
        super().__init__()
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟 RGB 相機 {index}：請確認相機權限，並在終端機分頁（非背景）執行。")
        self.size = (width, height)

    def read_once(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        return frame

    def close(self):
        self.cap.release()


class DepthCamera(Grabber):
    """輸出 uint16 深度圖，單位 mm，0 代表無效。"""

    name = "astra-depth"

    def __init__(self, width, height, fps):
        super().__init__()
        from pyorbbecsdk import Config, Context, OBFormat, OBLogLevel, OBSensorType, Pipeline

        Context.set_logger_level(OBLogLevel.NONE)
        self.pipe = Pipeline()
        profiles = self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        cfg = Config()
        cfg.enable_stream(profiles.get_video_stream_profile(width, height, OBFormat.Y11, fps))
        self.pipe.start(cfg)

    def read_once(self):
        frames = self.pipe.wait_for_frames(100)
        depth = frames.get_depth_frame() if frames else None
        if depth is None:
            return None
        data = np.frombuffer(depth.get_data(), dtype=np.uint16)
        data = data.reshape(depth.get_height(), depth.get_width())
        return (data * depth.get_depth_scale()).astype(np.uint16)

    def close(self):
        self.pipe.stop()


class AstraSource:
    """同時管理 RGB 與深度兩條擷取執行緒。"""

    def __init__(self, cfg):
        r, d = cfg["rgb"], cfg["depth"]
        self.rgb = RgbCamera(r["index"], r["width"], r["height"])
        try:
            self.depth = DepthCamera(d["width"], d["height"], d["fps"])
        except Exception:
            self.rgb.close()
            raise

    def start(self):
        self.rgb.start()
        self.depth.start()
        return self

    def stop(self):
        self.rgb.stop()
        self.depth.stop()

    def latest(self):
        """回傳 (rgb, rgb 時間戳, 深度 mm 或 None)。"""
        frame, stamp = self.rgb.latest()
        depth, _ = self.depth.latest()
        return frame, stamp, depth

    @property
    def error(self):
        return self.rgb.error or self.depth.error
