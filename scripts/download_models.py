"""下載執行所需模型與測試用範例圖（macOS / Linux / Windows 通用）。

    python scripts/download_models.py

  models/yolo11n-seg.pt             Ultralytics YOLO11 人體分割（AGPL-3.0）
  models/rtmpose-m_halpe26.onnx     RTMPose-m Halpe26 26 點（OpenMMLab MMPose，Apache-2.0）
  models/pose_landmarker_*.task     MediaPipe Pose Landmarker（Apache-2.0，替代骨架後端）
  tests/data/person.jpg             MediaPipe 官方範例圖（僅供測試）
"""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTMPOSE_ZIP = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip")
FILES = {
    "models/yolo11n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
    **{f"models/pose_landmarker_{v}.task":
       f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
       for v in ("lite", "full", "heavy")},
    "tests/data/person.jpg": "https://storage.googleapis.com/mediapipe-assets/pose.jpg",
}


def fetch(url, dest):
    """下載到 .part 再改名，中斷時不會留下不完整的檔案。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done * 100 // total}%", end="", flush=True)
    print()
    part.replace(dest)


def main(root=ROOT):
    root = Path(root)
    for rel, url in FILES.items():
        dest = root / rel
        if dest.exists():
            print(f"已存在：{rel}")
            continue
        fetch(url, dest)
        print(f"已下載：{rel}")

    onnx = root / "models" / "rtmpose-m_halpe26.onnx"
    if onnx.exists():
        print("已存在：models/rtmpose-m_halpe26.onnx")
    else:
        zpath = root / "models" / "rtmpose-m_halpe26.zip"
        fetch(RTMPOSE_ZIP, zpath)
        with zipfile.ZipFile(zpath) as z:
            member = next(n for n in z.namelist() if n.endswith("end2end.onnx"))
            with z.open(member) as src, open(onnx, "wb") as dst:
                shutil.copyfileobj(src, dst)
        zpath.unlink()
        print("已解壓：models/rtmpose-m_halpe26.onnx")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT)
