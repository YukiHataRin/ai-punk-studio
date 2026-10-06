"""把融合結果轉成 3D 視圖要畫的頂點與顏色（純 numpy，與繪圖框架無關）。

座標轉換：相機座標（x 右、y 下、z 前）→ OpenGL 世界座標（x 右、y 前、z 上）。
"""

from dataclasses import dataclass, field

import numpy as np

from .colors import hex_to_bgr, track_hex

DIM = 0.4  # 推估關節的亮度倍率


def to_gl(points):
    p = np.asarray(points, np.float32).reshape(-1, 3)
    return np.c_[p[:, 0], p[:, 2], -p[:, 1]]


def rgba(track_id, alpha=1.0, scale=1.0):
    b, g, r = hex_to_bgr(track_hex(track_id))
    return np.array([r / 255 * scale, g / 255 * scale, b / 255 * scale, alpha], np.float32)


@dataclass
class Scene:
    lines: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))       # 兩兩一組的線段端點
    line_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    joint_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    cloud_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    labels: list = field(default_factory=list)   # [(track_id, 位置, 文字)]
    ankle_heights: list = field(default_factory=list)  # 實測腳踝的 GL z，用來估地板


def build_scene(people, point_cloud=False):
    lines, lcol, joints, jcol, cloud, ccol, labels, ankles = [], [], [], [], [], [], [], []
    for person in people:
        tid = person.track_id
        bright, dim = rgba(tid), rgba(tid, 1.0, DIM)
        sk = person.skeleton
        if sk is not None:
            pts = to_gl(sk.points)
            fmt = sk.fmt
            for a, b in fmt.connections:
                if a in fmt.face or b in fmt.face:  # 臉部連線在 3D 裡太擠，省略
                    continue
                lines += [pts[a], pts[b]]
                c = bright if sk.measured[a] and sk.measured[b] else dim
                lcol += [c, c]
            body = np.array(fmt.body)
            joints.append(pts[body])
            jcol.append(np.where(sk.measured[body, None], bright, dim))
            ankles += [pts[j, 2] for j in fmt.ankles if sk.measured[j]]
            head = pts[fmt.head] + [0, 0, 0.25]
        elif person.centroid is not None:
            c = to_gl(person.centroid)[0]
            joints.append(c[None])
            jcol.append(bright[None])
            head = c + [0, 0, 0.5]
        else:
            continue
        dist = person.distance_text()
        labels.append((tid, head, f"ID {tid}" + (f" · {dist}" if dist else "")))
        if point_cloud and person.cloud is not None and len(person.cloud):
            cloud.append(to_gl(person.cloud))
            ccol.append(np.tile(rgba(tid, 0.55), (len(person.cloud), 1)))

    def stack(parts, width):
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros((0, width), np.float32)

    return Scene(
        np.array(lines, np.float32).reshape(-1, 3), np.array(lcol, np.float32).reshape(-1, 4),
        stack(joints, 3), stack(jcol, 4), stack(cloud, 3), stack(ccol, 4), labels, ankles,
    )
