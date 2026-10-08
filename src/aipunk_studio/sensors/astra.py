"""Orbbec Astra Pro：RGB 走 UVC（OpenCV / AVFoundation），深度走 pyorbbecsdk（OpenNI 協定）。"""

import time

import cv2
import numpy as np

from .base import CAPTURE_BACKEND, Grabber


class RgbCamera(Grabber):
    """UVC 攝影機（Astra Pro 的 RGB 鏡頭或一般攝影機）。開啟後太久沒有影格就報錯，不要讓介面一直空等。"""

    name = "rgb"
    STARTUP_TIMEOUT = 6.0

    def __init__(self, index, width, height):
        super().__init__()
        self.cap = cv2.VideoCapture(index, CAPTURE_BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟 RGB 相機 {index}：請確認攝影機已連接，以及相機權限"
                               "（macOS 需在已允許相機的終端機執行；Linux 需要 video 群組權限）。")
        self.size = (width, height)
        self._opened_at = time.monotonic()
        self._received = False

    def read_once(self):
        ok, frame = self.cap.read()
        if not ok:
            if not self._received and time.monotonic() - self._opened_at > self.STARTUP_TIMEOUT:
                raise RuntimeError("攝影機開啟了但沒有送出影像。可能被其他 App 占用，"
                                   "或 MacBook 螢幕蓋上時內建相機無法使用；請換一台攝影機或重新整理。")
            return None
        self._received = True
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        return frame

    def close(self):
        self.cap.release()


ASTRA_DEPTH_UNIT_MM = 10.0  # 原始值的單位；設定檔 [depth] unit_mm 可覆寫


class DepthCamera(Grabber):
    """輸出 uint16 深度圖，單位 mm，0 代表無效。

    unit_mm：原始值的單位。Astra Pro 經 pyorbbecsdk 1.3.2 讀到的值以 1 cm 為單位，
    但 SDK 的 get_depth_scale() 回報 1.0，所以不用它，改由設定指定。
    """

    name = "astra-depth"

    def __init__(self, width, height, fps, unit_mm=ASTRA_DEPTH_UNIT_MM):
        super().__init__()
        self.unit_mm = unit_mm
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
        return np.clip(data * self.unit_mm, 0, 65535).astype(np.uint16)

    def close(self):
        self.pipe.stop()


class DepthUnavailable(RuntimeError):
    """RGB 已開啟但深度相機開不起來（未接上、被其他程式占用等）。"""


class AstraSource:
    """同時管理 RGB 與深度兩條擷取執行緒。"""

    has_depth = True

    def __init__(self, cfg):
        r, d = cfg["rgb"], cfg["depth"]
        self.rgb = RgbCamera(r["index"], r["width"], r["height"])
        try:
            self.depth = DepthCamera(d["width"], d["height"], d["fps"], d.get("unit_mm", ASTRA_DEPTH_UNIT_MM))
        except Exception as error:
            self.rgb.close()
            raise DepthUnavailable(str(error)) from error

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
