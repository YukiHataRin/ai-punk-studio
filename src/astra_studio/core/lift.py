"""把 2D 關節點配上深度，得到 RGB 相機座標系下的公制 3D 骨架。

取樣規則（由嚴到寬）：
1. 有 YOLO 遮罩時，只採關節視窗內「屬於這個人遮罩」的深度值，別人或背景不會混進來；
2. 再用該人的基準深度做門檻（|z − ref| < depth_gate），濾掉對齊誤差造成的邊緣錯值；
3. 視窗內沒有遮罩像素時，把視窗放大到 3 倍再找一次（容忍遮罩略小於實際肢體）；
   仍然沒有就視為被遮擋，不取深度、改用推估——此時視窗裡的深度屬於擋在前面的東西。
沒有遮罩時只用門檻。

基準深度 ref：有遮罩時取整個遮罩的深度中位數，否則取軀幹（肩、髖）周圍的中位數。
深度量不到或可見度太低的關節，用 MediaPipe world landmarks 相對已知關節的位移補上。
"""

import numpy as np

from .geometry import Intrinsics
from .types import LEFT_HIP, RIGHT_HIP, PoseObservation, Skeleton3D

FALLBACK_DEPTH_M = 2.5  # 整個人都量不到深度時的假設距離
TORSO = (11, 12, 23, 24)  # 左右肩、左右髖
MIN_SAMPLES = 3


def _window(arr, u, v, win):
    h, w = arr.shape
    r = win // 2
    return arr[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]


def _masked_window(depth, mask, uv, win):
    """關節視窗內屬於遮罩的深度值；找不到時放大視窗一次，仍無則回傳 None。"""
    for w in (win, win * 3):
        inside = _window(depth, *uv, w)[_window(mask, *uv, w)]
        if inside.size:
            return inside
    return None


def _valid_m(depth_patch):
    return depth_patch[depth_patch > 0].astype(np.float32) / 1000.0


def reference_depth(reg_depth, uv, visible, win, mask=None):
    if mask is not None:
        vals = _valid_m(reg_depth[mask])
        if vals.size >= MIN_SAMPLES:
            return float(np.median(vals))
    vals = [_valid_m(_window(reg_depth, *uv[j], win * 2 + 1)) for j in TORSO if visible[j]]
    vals = np.concatenate(vals) if vals else np.empty(0)
    return float(np.median(vals)) if vals.size else 0.0


def lift_person(person: PoseObservation, reg_depth, reg_scale, rgb_k: Intrinsics, params, mask=None) -> Skeleton3D:
    """params：設定檔的 [skeleton] 區段；mask：與 reg_depth 同尺寸的 bool 遮罩（可省略）。"""
    n = len(person.pixels)
    win, gate = params["depth_window"], params["depth_gate"]
    uv = (person.pixels * reg_scale).astype(int)
    visible = person.visibility >= params["min_visibility"]
    ref = reference_depth(reg_depth, uv, visible, win, mask)

    pts = np.zeros((n, 3), np.float32)
    measured = np.zeros(n, bool)
    if ref > 0:
        for j in np.flatnonzero(visible):
            if mask is None:
                d = _window(reg_depth, *uv[j], win)
            else:
                d = _masked_window(reg_depth, mask, uv[j], win)
                if d is None:
                    continue
            vals = _valid_m(d)
            vals = vals[np.abs(vals - ref) < gate]
            if vals.size < MIN_SAMPLES:
                continue
            z = float(np.median(vals)) + params["joint_depth_offset"]
            pts[j] = rgb_k.backproject(*person.pixels[j], z)
            measured[j] = True

    # 用實測關節與 world landmarks 的平均位移，把剩下的關節補到同一個座標系
    if measured.any():
        offset = pts[measured].mean(0) - person.world[measured].mean(0)
    else:
        hip_px = person.pixels[[LEFT_HIP, RIGHT_HIP]].mean(0)
        offset = rgb_k.backproject(*hip_px, ref or FALLBACK_DEPTH_M)
    pts[~measured] = person.world[~measured] + offset

    distance = float(np.linalg.norm(pts[[LEFT_HIP, RIGHT_HIP]].mean(0)))
    return Skeleton3D(pts, measured, distance)
