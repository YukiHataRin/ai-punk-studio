"""MediaPipe Pose Landmarker（多人、VIDEO 模式）。

預設用 CPU（XNNPACK）。macOS 上 mediapipe 的 GPU delegate 每幀會洩漏輸入影像大小約 4 倍的
Metal 記憶體（0.10.35 與 1.0.1 都有，1280×720 約 14 MB/幀），30 fps 下不到一分鐘就會耗盡；
CPU 路徑不會洩漏，heavy 模型約 23 ms/幀，與 YOLO 平行執行時不影響幀率。
"""

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ..core.skeleton_format import MEDIAPIPE33
from ..core.types import PoseObservation


class PoseEstimator:
    def __init__(self, params):
        """params：設定檔的 [pose] 區段。"""
        self.gpu = params.get("delegate", "cpu") == "gpu"
        delegate = BaseOptions.Delegate.GPU if self.gpu else BaseOptions.Delegate.CPU
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=params["model"], delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=params["num_poses"],
            min_pose_detection_confidence=params["min_detection_confidence"],
            min_pose_presence_confidence=params["min_presence_confidence"],
            min_tracking_confidence=params["min_tracking_confidence"],
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1
        self.inference_ms = 0.0

    @property
    def device_label(self):
        return "Metal GPU" if self.gpu else "CPU"

    def __call__(self, bgr) -> list[PoseObservation]:
        h, w = bgr.shape[:2]
        if self.gpu:  # GPU 模式輸入必須是 SRGBA
            image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA))
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # VIDEO 模式時間戳必須遞增
        self._last_ts = ts
        started = time.perf_counter()
        result = self.landmarker.detect_for_video(image, ts)
        self.inference_ms = (time.perf_counter() - started) * 1000

        return [PoseObservation(
            pixels=np.array([[lm.x * w, lm.y * h] for lm in lms], np.float32),
            visibility=np.array([lm.visibility or 0.0 for lm in lms], np.float32),
            world=np.array([[lm.x, lm.y, lm.z] for lm in wlms], np.float32),
            fmt=MEDIAPIPE33,
        ) for lms, wlms in zip(result.pose_landmarks, result.pose_world_landmarks)]

    def close(self):
        self.landmarker.close()
