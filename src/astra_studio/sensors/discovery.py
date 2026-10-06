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

    backend = {"darwin": cv2.CAP_AVFOUNDATION, "win32": cv2.CAP_MSMF}.get(sys.platform, cv2.CAP_V4L2)
    return [Camera(item.name or "Camera", item.index, item.backend, item.path or f"{item.backend}:{item.index}")
            for item in enumerate_cameras(backend)]


def discover_cameras():
    result = subprocess.run(
        [sys.executable, "-m", "astra_studio.sensors.discovery"],
        cwd=PROJECT_ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True, timeout=10, check=True,
    )
    return [Camera(**item) for item in json.loads(result.stdout)]


if __name__ == "__main__":
    print(json.dumps([asdict(camera) for camera in enumerate_devices()]))
