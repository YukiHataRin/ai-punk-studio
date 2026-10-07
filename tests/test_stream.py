"""WebSocket 串流與 headless 測試（不需要相機）。"""

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

from astra_studio.config import PROJECT_ROOT, load_config
from astra_studio.core.skeleton_format import HALPE26
from astra_studio.core.types import PoseObservation, SegmentationResult, SegmentedPerson, Skeleton3D, TrackedPerson
from astra_studio.io.stream import StreamServer, frame_message, mask_contours
from astra_studio.pipeline.pipeline import FrameOutput
from astra_studio.pipeline.runner import CaptureRunner

CFG = load_config()


def fake_output():
    mask = np.zeros((720, 1280), bool)
    mask[200:600, 400:600] = True
    seg = SegmentedPerson(mask, np.array([400, 200, 600, 600], float), 0.9, 5)
    pose = PoseObservation(np.full((26, 2), 500, np.float32), np.ones(26, np.float32), None, HALPE26)
    pts = np.tile([0.0, 0.0, 2.0], (26, 1)).astype(np.float32)
    sk = Skeleton3D(pts, np.ones(26, bool), 2.0, HALPE26)
    person = TrackedPerson(5, seg, pose, sk, pts[0])
    return FrameOutput(np.zeros((720, 1280, 3), np.uint8), np.zeros((360, 640), np.uint16),
                       SegmentationResult([seg]), [person], {})


def recv_type(ws, kind, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = json.loads(ws.recv(timeout=timeout))
        if msg["type"] == kind:
            return msg
    raise AssertionError(f"沒收到 {kind}")


class MessageTest(unittest.TestCase):
    def test_frame_message(self):
        msg = frame_message(fake_output(), {"fps": 29.94, "has_depth": True}, 7, 1.23456, {"contours": True})
        json.dumps(msg)  # 必須可序列化
        self.assertEqual((msg["type"], msg["frame"], msg["fps"], msg["t"]), ("frame", 7, 29.9, 1.2346))
        p = msg["people"][0]
        self.assertEqual((p["id"], p["format"], len(p["joints"])), (5, "halpe26", 26))
        poly = np.array(p["contour"][0])
        self.assertTrue((poly[:, 0] >= 399).all() and (poly[:, 0] <= 600).all())
        self.assertEqual(len(poly), 4)  # 矩形遮罩簡化後剩 4 個角
        lean = frame_message(fake_output(), {}, 0, 0, {"contours": False, "pixels": False})
        self.assertNotIn("contour", lean["people"][0])
        self.assertNotIn("pixels", lean["people"][0])

    def test_tiny_blobs_are_dropped(self):
        mask = np.zeros((100, 100), bool)
        mask[1:4, 1:4] = True
        self.assertEqual(mask_contours(mask), [])


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.server = StreamServer("127.0.0.1", 0, {"max_fps": 0}).start()

    def tearDown(self):
        self.server.stop()

    def test_hello_frames_and_ping(self):
        self.server.start_session(CFG, has_depth=False)
        with connect(self.server.url) as ws:
            hello = recv_type(ws, "hello")
            self.assertFalse(hello["has_depth"])
            self.assertEqual(len(hello["formats"]["halpe26"]["names"]), 26)
            deadline = time.monotonic() + 2
            while self.server.client_count == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.server.publish(fake_output(), {"fps": 30.0, "has_depth": False})
            frame = recv_type(ws, "frame")
            self.assertEqual(frame["people"][0]["id"], 5)
            ws.send(json.dumps({"type": "ping"}))
            self.assertEqual(recv_type(ws, "pong")["type"], "pong")

    def test_burst_arrives_in_order_ending_with_latest(self):
        """連續推送時：順序不亂、不重複，且最後一則一定是最新幀（傳送受阻時中間的舊幀才會被跳過）。"""
        self.server.start_session(CFG, has_depth=True)
        with connect(self.server.url) as ws:
            recv_type(ws, "hello")
            while self.server.client_count == 0:
                time.sleep(0.01)
            for _ in range(200):  # 用戶端還沒讀，伺服器連續推 200 幀
                self.server.publish(fake_output(), {"fps": 30.0})
            time.sleep(0.3)
            frames = []
            try:
                while True:
                    frames.append(json.loads(ws.recv(timeout=0.3))["frame"])
            except TimeoutError:
                pass
        self.assertEqual(frames, sorted(set(frames)))
        self.assertEqual(frames[-1], 199)

    def test_port_in_use(self):
        with self.assertRaises(RuntimeError):
            StreamServer("127.0.0.1", self.server.port, {}).start()


class FakeSource:
    has_depth = False
    error = None

    def __init__(self, cfg):
        self.n = 0

    def start(self):
        return self

    def stop(self):
        pass

    def latest(self):
        self.n += 1
        time.sleep(0.01)
        return np.zeros((720, 1280, 3), np.uint8), float(self.n), None


class FakePipeline:
    segmenter = None
    pose_label = "Fake"

    def __init__(self, cfg, segmentation=True):
        pass

    def process(self, frame, depth, stamp):
        return fake_output()

    def close(self):
        pass


class RunnerStreamTest(unittest.TestCase):
    def test_runner_publishes_to_clients(self):
        server = StreamServer("127.0.0.1", 0, {"max_fps": 0}).start()
        runner = CaptureRunner(CFG, source_factory=FakeSource, pipeline_factory=FakePipeline, publisher=server)
        thread = threading.Thread(target=runner.run)
        thread.start()
        try:
            with connect(server.url) as ws:
                hello = recv_type(ws, "hello")
                self.assertFalse(hello["has_depth"])  # 來自 FakeSource.has_depth
                frame = recv_type(ws, "frame")
                self.assertFalse(frame["has_depth"])
                self.assertEqual(frame["people"][0]["id"], 5)
        finally:
            runner.stop()
            thread.join(5)
            server.stop()
        self.assertFalse(thread.is_alive())


@unittest.skipUnless(Path(CFG["segmentation"]["model"]).exists() and Path(CFG["pose"]["rtm_model"]).exists(), "缺模型")
class HeadlessCliTest(unittest.TestCase):
    def test_headless_playback_runs_and_exits(self):
        from astra_studio.io.recorder import SessionRecorder
        with tempfile.TemporaryDirectory() as tmp:
            rec = SessionRecorder(CFG, root=tmp, with_images=True)
            for i in range(30):
                rec.write(np.full((720, 1280, 3), 120, np.uint8), np.full((480, 640), 2000, np.uint16), fake_output(), i / 30)
            path = rec.close()
            result = subprocess.run(
                [sys.executable, "-m", "astra_studio", "--headless", "--play", str(path), "--ws-port", "0"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WebSocket 串流：ws://127.0.0.1:", result.stdout)
        self.assertIn("播放完畢", result.stdout)
        self.assertIn("結束，共處理", result.stdout)


if __name__ == "__main__":
    unittest.main()
