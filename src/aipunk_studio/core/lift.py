"""把 2D 關節點配上深度，得到 RGB 相機座標系下的公制 3D 骨架。

取樣規則（由嚴到寬）：
1. 有 YOLO 遮罩時，只採關節視窗內「屬於這個人遮罩」的深度值，別人或背景不會混進來；
2. 再用該人的基準深度做門檻（|z − ref| < depth_gate），濾掉對齊誤差造成的邊緣錯值；
3. 視窗內沒有遮罩像素時，把視窗放大到 3 倍再找一次（容忍遮罩略小於實際肢體）；
   仍然沒有就視為被遮擋，不取深度、改用推估——此時視窗裡的深度屬於擋在前面的東西。
沒有遮罩時只用門檻。

基準深度 ref：有遮罩時取整個遮罩的深度中位數，否則取軀幹（肩、髖）周圍的中位數。

整個人都量不到深度時（一般攝影機、或人在深度範圍外），用身體尺寸估計距離：
針孔模型 距離 ≈ 焦距 × 實際長度 ÷ 像素長度，優先用軀幹長（肩中點到髖中點，轉身時幾乎不變），
看不到髖時改用肩寬；兩者都看不到才用固定的 FALLBACK_DEPTH_M。

量不到深度的關節（推估）：
- 有 world landmarks（MediaPipe）：把 world 骨架平移到實測關節上補齊；
- 沒有（RTMPose）：2D 位置照用，深度取這個人實測關節的中位數（只看身體關節，實測少於 MIN_PLANE_JOINTS 個就用 ref），
  也就是假設該關節落在人體所在的深度平面上。
"""

import numpy as np

from .geometry import Intrinsics
from .types import PoseObservation, Skeleton3D

FALLBACK_DEPTH_M = 2.5  # 連身體尺寸都看不到時的假設距離
MIN_EST_M, MAX_EST_M = 0.3, 15.0
MIN_SAMPLES = 3
MIN_PLANE_JOINTS = 3  # 推估關節的深度平面：至少這麼多實測關節才用它們的中位數，否則用基準深度


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


def reference_depth(reg_depth, uv, visible, win, torso, mask=None):
    if mask is not None:
        vals = _valid_m(reg_depth[mask])
        if vals.size >= MIN_SAMPLES:
            return float(np.median(vals))
    vals = [_valid_m(_window(reg_depth, *uv[j], win * 2 + 1)) for j in torso if visible[j]]
    vals = np.concatenate(vals) if vals else np.empty(0)
    return float(np.median(vals)) if vals.size else 0.0


def body_mask(fmt):
    """可以當深度錨點的關節：排除鼻子、臉部點與頭頂。"""
    m = np.zeros(fmt.size, bool)
    m[list(fmt.body)] = True
    m[[fmt.head, fmt.names.index("nose")]] = False
    return m


def estimate_distance(person: PoseObservation, rgb_k: Intrinsics, params):
    """從身體尺寸估計人到相機的距離（公尺）；看不到軀幹或肩膀時回傳 None。"""
    fmt, px = person.fmt, person.pixels
    ok = person.visibility >= params["min_visibility"]
    ls, rs = fmt.shoulders
    lh, rh = fmt.hips
    if ok[[ls, rs, lh, rh]].all():
        torso_px = float(np.linalg.norm(px[[ls, rs]].mean(0) - px[[lh, rh]].mean(0)))
        if torso_px > 5:
            return float(np.clip(rgb_k.fy * params["est_torso_m"] / torso_px, MIN_EST_M, MAX_EST_M))
    if ok[[ls, rs]].all():
        shoulder_px = float(np.linalg.norm(px[ls] - px[rs]))
        if shoulder_px > 5:
            return float(np.clip(rgb_k.fx * params["est_shoulder_m"] / shoulder_px, MIN_EST_M, MAX_EST_M))
    return None


def lift_person(person: PoseObservation, reg_depth, reg_scale, rgb_k: Intrinsics, params, mask=None) -> Skeleton3D:
    """params：設定檔的 [skeleton] 區段；mask：與 reg_depth 同尺寸的 bool 遮罩（可省略）。"""
    n = len(person.pixels)
    win, gate = params["depth_window"], params["depth_gate"]
    uv = (person.pixels * reg_scale).astype(int)
    visible = person.visibility >= params["min_visibility"]
    fmt = person.fmt
    ref = reference_depth(reg_depth, uv, visible, win, fmt.torso, mask)

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

    missing = ~measured
    if not measured.any() and not ref:
        ref = estimate_distance(person, rgb_k, params) or FALLBACK_DEPTH_M
    if missing.any():
        if person.world is not None:
            # 用實測關節與 world landmarks 的平均位移，把剩下的關節補到同一個座標系
            if measured.any():
                offset = pts[measured].mean(0) - person.world[measured].mean(0)
            else:
                hip_px = person.pixels[list(fmt.hips)].mean(0)
                offset = rgb_k.backproject(*hip_px, ref)
            pts[missing] = person.world[missing] + offset
        else:
            # 深度平面只看身體關節：臉部與頭頂擠在頭部邊緣，常量到後方背景；
            # 實測的身體關節太少時中位數不可靠，改用整個遮罩的基準深度，否則一兩個錯值就會把整副骨架推走
            anchors = measured & body_mask(fmt)
            z = float(np.median(pts[anchors, 2])) if anchors.sum() >= MIN_PLANE_JOINTS or not ref else ref
            pts[missing] = rgb_k.backproject(person.pixels[missing, 0], person.pixels[missing, 1], z)

    sk = Skeleton3D(pts, measured, 0.0, fmt)
    sk.distance = float(np.linalg.norm(sk.hip_center))
    return sk
