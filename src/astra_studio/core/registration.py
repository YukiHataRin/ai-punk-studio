"""把深度圖投影到 RGB 相機視角（軟體 D2C）。

Astra Pro 的硬體對齊在 macOS 上無法使用，這裡用內參 + 外參自己做：
深度像素 → 深度相機 3D 點 → RGB 相機 3D 點 → RGB 像素。
"""

import numpy as np

from .geometry import Intrinsics, rotation_matrix


class DepthToRgb:
    def __init__(self, cfg, scale=0.5, stride=2):
        """scale：輸出深度圖相對 RGB 解析度的比例；stride：深度圖取樣間隔（加速用）。"""
        d, c = cfg["depth"], cfg["rgb"]
        self.scale = scale
        self.out_w, self.out_h = int(c["width"] * scale), int(c["height"] * scale)
        self.rgb_k = Intrinsics.from_config(c, scale)
        self.min_mm, self.max_mm = d["min_mm"], d["max_mm"]
        self.stride = stride

        v, u = np.mgrid[0:d["height"]:stride, 0:d["width"]:stride]
        self._rx = ((u - d["cx"]) / d["fx"]).ravel()
        self._ry = ((v - d["cy"]) / d["fy"]).ravel()
        self.R = rotation_matrix(cfg["extrinsics"]["rotation_deg"])
        self.t = np.array(cfg["extrinsics"]["translation"], dtype=np.float64)

    def empty(self):
        return np.zeros((self.out_h, self.out_w), np.uint16)

    def __call__(self, depth_mm):
        """回傳 RGB 視角的深度圖（uint16 mm，尺寸 = RGB × scale）。"""
        z = depth_mm[:: self.stride, :: self.stride].ravel().astype(np.float64) / 1000.0
        keep = (z * 1000 > self.min_mm) & (z * 1000 < self.max_mm)
        z = z[keep]
        pts = np.stack([self._rx[keep] * z, self._ry[keep] * z, z])
        pts = self.R @ pts + self.t[:, None]

        k = self.rgb_k
        zc = pts[2]
        u = np.round(pts[0] / zc * k.fx + k.cx).astype(np.int32)
        v = np.round(pts[1] / zc * k.fy + k.cy).astype(np.int32)
        ok = (zc > 0) & (u >= 0) & (u < self.out_w) & (v >= 0) & (v < self.out_h)
        u, v, zc = u[ok], v[ok], zc[ok]

        # 由遠到近寫入，近的點覆蓋遠的點（簡易 z-buffer）
        order = np.argsort(-zc)
        out = self.empty()
        out[v[order], u[order]] = (zc[order] * 1000).astype(np.uint16)
        return out
