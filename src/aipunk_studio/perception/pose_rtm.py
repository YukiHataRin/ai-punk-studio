"""RTMPose（Halpe26，26 點含腳）由上而下的逐人骨架估計：拿 YOLO 的人框裁切，每人各跑一次。

多人時比 MediaPipe 穩定：骨架直接屬於該框的 track ID，不需要事後配對，也不會在人與人之間跳動；
每人都以完整解析度（256×192）估計，遠處的人也較準。

前後處理與 rtmlib（Apache-2.0）的 RTMPose 一致：框放大 1.25 倍、補成 192:256、仿射裁切、
以 mean/std 正規化（直接用 BGR，與 rtmlib 相同）、SimCC 取最大值（split ratio 2，分數取兩軸最大值平均）。

推論用 onnxruntime，rtm_provider = auto 時依平台選：macOS 用 CoreML（每人約 4 ms，Apple M5）；
裝了 onnxruntime-gpu 且有 NVIDIA GPU 時用 CUDA；其餘用 CPU（每人約 10 ms）。
CoreML 在 batch 大小改變時會失敗，所以固定 batch = 1 逐人執行；GPU 後端出錯時自動改用 CPU。
"""

import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

from ..core.skeleton_format import HALPE26
from ..core.types import PoseObservation

# 主要靠 aipunk_studio/__init__.py 設定的 ORT_DISABLE_TELEMETRY=1（匯入前生效）；這裡再保險關一次
ort.disable_telemetry_events()

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
PADDING = 1.25
SIMCC_SPLIT = 2.0


def box_center_scale(box, aspect):
    """xyxy 框 → 中心與裁切尺寸（放大 PADDING 倍並補成模型長寬比 w/h）。"""
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    w, h = (x2 - x1) * PADDING, (y2 - y1) * PADDING
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], np.float32)


def crop_matrix(center, scale, out_w, out_h):
    """把以 center 為中心、大小 scale 的區域映射到 (out_w, out_h) 的仿射矩陣（無旋轉）。"""
    sx, sy = out_w / scale[0], out_h / scale[1]
    return np.array([[sx, 0, out_w / 2 - center[0] * sx],
                     [0, sy, out_h / 2 - center[1] * sy]], np.float32)


def decode_simcc(simcc_x, simcc_y):
    """(K, Wx)、(K, Hy) → 模型輸入座標 (K, 2) 與分數 (K,)。"""
    locs = np.stack([simcc_x.argmax(1), simcc_y.argmax(1)], axis=1).astype(np.float32) / SIMCC_SPLIT
    scores = 0.5 * (simcc_x.max(1) + simcc_y.max(1))
    return locs, scores


def resolve_provider(name):
    """auto → 依平台與已安裝的 onnxruntime 選擇 coreml / cuda / cpu。"""
    if name != "auto":
        return name
    available = ort.get_available_providers()
    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        return "coreml"
    if "CUDAExecutionProvider" in available:
        return "cuda"
    return "cpu"


class RTMPoseEstimator:
    fmt = HALPE26

    def __init__(self, params):
        """params：設定檔的 [pose] 區段（使用 rtm_model、rtm_provider、num_poses）。"""
        self.model_path = params["rtm_model"]
        self.max_people = params.get("num_poses", 4)
        self.inference_ms = 0.0
        self._open(resolve_provider(params.get("rtm_provider", "auto")))

    def _open(self, provider):
        accel = {"coreml": "CoreMLExecutionProvider", "cuda": "CUDAExecutionProvider"}.get(provider)
        providers = [accel, "CPUExecutionProvider"] if accel else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        active = self.session.get_providers()[0]
        self.provider = provider if active == accel else "cpu"  # GPU 後端載入失敗時 onnxruntime 會默默退回 CPU
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = inp.shape[2], inp.shape[3]
        self.output_names = [o.name for o in self.session.get_outputs()]

    @property
    def device_label(self):
        return {"coreml": "CoreML", "cuda": "CUDA"}.get(self.provider, "CPU")

    def _infer(self, blob):
        try:
            return self.session.run(self.output_names, {self.input_name: blob})
        except Exception:
            if self.provider == "cpu":
                raise
            self._open("cpu")  # CoreML 偶爾不支援某些輸入，改用 CPU 繼續
            return self.session.run(self.output_names, {self.input_name: blob})

    def estimate_one(self, bgr, box):
        center, scale = box_center_scale(box, self.in_w / self.in_h)
        M = crop_matrix(center, scale, self.in_w, self.in_h)
        crop = cv2.warpAffine(bgr, M, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR)
        blob = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self._infer(np.ascontiguousarray(blob))
        locs, scores = decode_simcc(simcc_x[0], simcc_y[0])
        pixels = locs / [self.in_w, self.in_h] * scale + center - scale / 2  # 模型座標 → 原圖像素
        return PoseObservation(pixels.astype(np.float32), scores.astype(np.float32), None, HALPE26)

    def __call__(self, bgr, boxes):
        """boxes：xyxy 框列表。回傳與 boxes 等長的 PoseObservation；超過 max_people 的（框較小者）為 None。"""
        started = time.perf_counter()
        order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        keep = set(order[:self.max_people])
        out = [self.estimate_one(bgr, b) if i in keep else None for i, b in enumerate(boxes)]
        self.inference_ms = (time.perf_counter() - started) * 1000
        return out

    def close(self):
        self.session = None
