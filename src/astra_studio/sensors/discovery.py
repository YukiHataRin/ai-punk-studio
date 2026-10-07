"""依名稱列舉相機（取自 Human Mask Studio 的 studio/cameras.py）。

在子程序中列舉，熱插拔後 AVFoundation 的裝置清單才會更新。
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass

import cv2

from ..config import PROJECT_ROOT


@dataclass(frozen=True)
class Camera:
    name: str
    index: int
    backend: int
    uid: str

    @property
    def is_astra(self):
        return "astra" in self.name.lower()


def enumerate_devices():
    from cv2_enumerate_cameras import enumerate_cameras

    from .base import CAPTURE_BACKEND
    backend = CAPTURE_BACKEND
    return [Camera(item.name or "Camera", item.index, item.backend, item.path or f"{item.backend}:{item.index}")
            for item in enumerate_cameras(backend)]


def find_astra_rgb_index(default):
    """在相機清單中找 Astra Pro 的 RGB 鏡頭；找不到就用 default。插拔其他攝影機後編號會變，所以要依名稱找。"""
    try:
        for camera in discover_cameras():
            if camera.is_astra:
                return camera.index
    except Exception:
        pass
    return default


def select_camera(cameras, want=None):
    """want：名稱、裝置 ID 或編號（字串）；沒指定時優先 Astra Pro，再來第一台。找不到回傳 None。"""
    if want is not None:
        return next((c for c in cameras if want in (c.name, c.uid, str(c.index))), None)
    return next((c for c in cameras if c.is_astra), cameras[0] if cameras else None)


def apply_camera(cfg, camera):
    """把選到的攝影機寫進設定（就地修改）：cfg["source"]，以及一般攝影機的視角推算內參。"""
    cfg["source"] = {"kind": "astra" if camera.is_astra else "rgb", "index": camera.index, "name": camera.name}
    if not camera.is_astra:  # 一般攝影機沒有出廠內參，用水平視角推算
        from .webcam import intrinsics_from_fov
        r = cfg["rgb"]
        r.update(intrinsics_from_fov(r["width"], r["height"], cfg["webcam"]["hfov_deg"]))
    return cfg


def discover_cameras():
    result = subprocess.run(
        [sys.executable, "-m", "astra_studio.sensors.discovery"],
        cwd=PROJECT_ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True, timeout=10, check=True,
    )
    return [Camera(**item) for item in json.loads(result.stdout)]


if __name__ == "__main__":
    print(json.dumps([asdict(camera) for camera in enumerate_devices()]))
