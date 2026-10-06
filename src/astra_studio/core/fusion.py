"""融合 MediaPipe 骨架與 YOLO 人體分割。

配對分數 = 骨架可見關節落在該人遮罩內的比例。分數由高到低貪婪配對，
低於 min_score 的不配。人數最多個位數，貪婪法已足夠，不需要匈牙利演算法。
"""

import cv2
import numpy as np

from .geometry import Intrinsics
from .types import PoseObservation, SegmentedPerson

BODY = np.r_[0, 11:33]  # 鼻子 + 身體關節；臉部其他點都擠在頭部，不提供額外資訊
MIN_VISIBILITY = 0.5


def match_score(pose: PoseObservation, mask: np.ndarray):
    vis = BODY[pose.visibility[BODY] >= MIN_VISIBILITY]
    if vis.size == 0:
        return 0.0
    h, w = mask.shape
    u = np.clip(pose.pixels[vis, 0].astype(int), 0, w - 1)
    v = np.clip(pose.pixels[vis, 1].astype(int), 0, h - 1)
    return float(mask[v, u].mean())


def associate(poses, segments, min_score=0.3):
    """回傳 {pose 索引: segment 索引}。"""
    scores = [(match_score(p, s.mask), i, j) for i, p in enumerate(poses) for j, s in enumerate(segments)]
    pairs, used_p, used_s = {}, set(), set()
    for score, i, j in sorted(scores, reverse=True):
        if score < min_score:
            break
        if i in used_p or j in used_s:
            continue
        pairs[i] = j
        used_p.add(i)
        used_s.add(j)
    return pairs


def depth_mask(segment: SegmentedPerson, shape, erode_px=2):
    """把原圖解析度的遮罩縮到對齊深度圖的解析度，並內縮幾個像素避開邊緣（對齊誤差在邊緣最明顯）。"""
    m = cv2.resize(segment.mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    return m.astype(bool)


def mask_points(mask_small, reg_depth, reg_k: Intrinsics, stride=3, max_points=3000):
    """遮罩內的深度像素 → 3D 點雲（公尺，RGB 相機座標系），給 3D 視圖用。"""
    m = np.zeros_like(mask_small)
    m[::stride, ::stride] = mask_small[::stride, ::stride]
    v, u = np.nonzero(m & (reg_depth > 0))
    if u.size > max_points:
        keep = np.linspace(0, u.size - 1, max_points).astype(int)
        u, v = u[keep], v[keep]
    return reg_k.backproject(u, v, reg_depth[v, u] / 1000.0).astype(np.float32)


def mask_centroid(mask_small, reg_depth, reg_k: Intrinsics, min_pixels=20):
    """遮罩內深度中位數 + 遮罩重心像素 → 3D 位置（公尺，RGB 相機座標系）。量不到時回傳 None。

    reg_k 是對齊深度圖解析度下的 RGB 內參。
    """
    vals = reg_depth[mask_small]
    vals = vals[vals > 0]
    if vals.size < min_pixels:
        return None
    v, u = np.nonzero(mask_small)
    z = float(np.median(vals)) / 1000.0
    return reg_k.backproject(u.mean(), v.mean(), z).astype(np.float32)
