"""一般攝影機（只有 RGB、沒有深度），介面與 AstraSource 相同；latest() 的深度永遠是 None，
下游會改用身體尺寸估計 3D（見 core/lift.py）。"""

import math

from .astra import RgbCamera


def intrinsics_from_fov(width, height, hfov_deg):
    """沒有出廠校正時，以水平視角推算針孔內參（方形像素、主點在畫面中心）。"""
    fx = width / (2 * math.tan(math.radians(hfov_deg) / 2))
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


class WebcamSource:
    has_depth = False

    def __init__(self, cfg, index=None):
        r = cfg["rgb"]
        self.rgb = RgbCamera(r["index"] if index is None else index, r["width"], r["height"])

    def start(self):
        self.rgb.start()
        return self

    def stop(self):
        self.rgb.stop()

    def latest(self):
        frame, stamp = self.rgb.latest()
        return frame, stamp, None

    @property
    def error(self):
        return self.rgb.error
