"""相機內參與座標轉換。座標系：x 向右、y 向下、z 向前，單位公尺。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_config(cls, section, scale=1.0):
        return cls(section["fx"] * scale, section["fy"] * scale, section["cx"] * scale, section["cy"] * scale)

    def backproject(self, u, v, z):
        """像素 (u, v) + 深度 z（公尺）→ 3D 點。u、v、z 可為純量或陣列。"""
        u, v, z = (np.asarray(a, np.float64) for a in (u, v, z))
        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z
        return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def rotation_matrix(deg_xyz):
    rx, ry, rz = np.deg2rad(deg_xyz)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
