"""錄製、回放、CSV 匯出測試（合成資料，不需要相機與模型）。"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from aipunk_studio.config import load_config
from aipunk_studio.core.types import PoseObservation, SegmentationResult, Skeleton3D, TrackedPerson
from aipunk_studio.io.recorder import SessionRecorder, read_skeleton
from aipunk_studio.pipeline.pipeline import FrameOutput
from aipunk_studio.sensors.playback import PlaybackSource

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_skeleton_csv import export, export_metrics  # noqa: E402

CFG = load_config()
N = 12


def synthetic(i):
    frame = np.full((720, 1280, 3), (i * 20) % 255, np.uint8)
    depth = np.full((480, 640), 1000 + i, np.uint16)
    pts = np.c_[np.zeros(33), np.linspace(-0.8, 0.9, 33), np.full(33, 2.0)].astype(np.float32)
    pose = PoseObservation(np.zeros((33, 2), np.float32), np.ones(33, np.float32), np.zeros((33, 3), np.float32))
    people = [TrackedPerson(3, None, pose, Skeleton3D(pts, np.ones(33, bool), 2.0), pts.mean(0),
                            metrics={"energy": 0.5 + i, "height": None}),
              TrackedPerson(5, None, None, None, np.array([1.0, 0.0, 3.0], np.float32))]  # 只有遮罩位置
    return frame, depth, FrameOutput(frame, depth, SegmentationResult(), people, {})


class RecorderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rec = SessionRecorder(CFG, root=cls.tmp.name, with_images=True)
        for i in range(N):
            frame, depth, out = synthetic(i)
            rec.write(frame, depth if i != 4 else None, out, 100.0 + i / 30)  # 第 4 幀沒有深度
        cls.dir = rec.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_files(self):
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["with_images"])
        self.assertEqual(meta["rgb"]["fx"], CFG["rgb"]["fx"])
        self.assertTrue((self.dir / "rgb.mp4").stat().st_size > 0)
        self.assertEqual(len(list((self.dir / "depth").glob("*.png"))), N - 1)

    def test_depth_is_lossless(self):
        d = cv2.imread(str(self.dir / "depth" / "000007.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(d.dtype, np.uint16)
        self.assertTrue((d == 1007).all())

    def test_skeleton_jsonl(self):
        recs = list(read_skeleton(self.dir / "skeleton.jsonl"))
        self.assertEqual(len(recs), N)
        self.assertAlmostEqual(recs[3]["t"], 0.1, places=3)
        p3, p5 = recs[0]["people"]
        self.assertEqual(len(p3["joints"]), 33)
        self.assertEqual(p3["format"], "mediapipe33")
        self.assertEqual(p3["metric_joints_measured"], 13)
        self.assertTrue(p3["distance_measured"])
        self.assertNotIn("joints", p5)
        self.assertEqual(p5["centroid"], [1.0, 0.0, 3.0])

    def test_csv_export(self):
        path, rows = export(self.dir)
        self.assertEqual(rows, N * 33)  # 只有 ID 3 有骨架
        header = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "frame,t,id,joint,joint_name,x,y,z,measured")
        path, rows = export_metrics(self.dir)
        self.assertEqual(rows, N)  # ID 5 只有遮罩，沒有指標
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("frame,t,id,distance,distance_measured,energy,"))
        self.assertTrue(lines[2].startswith("1,0.0333,3,2.0,1,1.5,"))

    def test_playback(self):
        src = PlaybackSource(self.dir).start()
        try:
            deadline = time.monotonic() + 5
            while not src.finished and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            src.stop()
        self.assertTrue(src.finished)
        self.assertIsNone(src.error)
        frame, _, depth = src.latest()
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertTrue((depth == 1000 + N - 1).all())
        self.assertEqual(len(src), N)

    def test_skeleton_only_recording_cannot_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = SessionRecorder(CFG, root=tmp, with_images=False)
            frame, depth, out = synthetic(0)
            rec.write(frame, depth, out, 0.0)
            path = rec.close()
            self.assertFalse((path / "rgb.mp4").exists())
            with self.assertRaises(RuntimeError):
                PlaybackSource(path)


if __name__ == "__main__":
    unittest.main()


class CameraTimeoutTest(unittest.TestCase):
    def test_camera_without_frames_reports_error(self):
        """攝影機能開啟卻不送影像（例如 MacBook 蓋上螢幕時的內建相機）時，要報錯而不是讓介面一直空等。"""
        from unittest import mock
        from aipunk_studio.sensors import astra

        class DeadCapture:
            def __init__(self, *args): pass
            def set(self, *args): pass
            def isOpened(self): return True
            def read(self):
                time.sleep(0.005)
                return False, None
            def release(self): pass

        with mock.patch.object(astra.cv2, "VideoCapture", DeadCapture):
            cam = astra.RgbCamera(1, 1280, 720)
            cam.STARTUP_TIMEOUT = 0.2
            cam.start()
            deadline = time.monotonic() + 3
            while cam.error is None and time.monotonic() < deadline:
                time.sleep(0.02)
            cam.stop()
        self.assertIn("沒有送出影像", cam.error)
