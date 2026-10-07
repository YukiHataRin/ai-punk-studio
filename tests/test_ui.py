"""Qt 介面測試：offscreen 平台 + 假 worker，不需要相機與模型。"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from astra_studio.config import load_config
from astra_studio.core.skeleton_format import MEDIAPIPE33
from astra_studio.core.types import PoseObservation, SegmentationResult, SegmentedPerson, Skeleton3D, TrackedPerson
from astra_studio.pipeline.pipeline import FrameOutput
from astra_studio.render.scene3d import build_scene, to_gl
from astra_studio.ui.main_window import MainWindow
from astra_studio.ui.theme import apply_theme

APP = QApplication.instance() or QApplication([])
apply_theme(APP)
CFG = load_config()


def fake_output(n_people=2):
    frame = np.full((720, 1280, 3), 40, np.uint8)
    people, segs = [], []
    for i in range(n_people):
        mask = np.zeros((720, 1280), bool)
        mask[200:600, 300 + 400 * i:450 + 400 * i] = True
        seg = SegmentedPerson(mask, np.array([300 + 400 * i, 200, 450 + 400 * i, 600]), 0.9, i + 1)
        pixels = np.c_[np.linspace(320, 430, 33) + 400 * i, np.linspace(220, 580, 33)].astype(np.float32)
        pose = PoseObservation(pixels, np.ones(33, np.float32), np.zeros((33, 3), np.float32))
        pts = np.c_[np.zeros(33), np.linspace(-0.8, 0.9, 33), np.full(33, 2.0 + i)].astype(np.float32)
        measured = np.arange(33) % 3 != 0
        sk = Skeleton3D(pts, measured, 2.0 + i)
        segs.append(seg)
        people.append(TrackedPerson(i + 1, seg, pose, sk, pts.mean(0), pts))
    people.append(TrackedPerson(9, segs[0], None, None, np.array([0.5, 0.0, 3.0], np.float32)))  # 只有遮罩
    return FrameOutput(frame, np.full((360, 640), 2000, np.uint16), SegmentationResult(segs, 20.0), people,
                       {"segmentation": 20.0, "pose": 10.0, "fusion": 1.2, "total": 25.0})


class FakeWorker(QObject):
    status = Signal(str)
    failed = Signal(str)
    ready = Signal(str)
    finished = Signal()

    recording = Signal(str, bool)

    def __init__(self, cfg, segmentation, parent=None, source_factory=None, publisher=None):
        super().__init__(parent)
        self.cfg, self.updates, self.output = cfg, {}, (fake_output(), {"fps": 29.5, "depth": True})
        self.runner = type("Runner", (), {"publisher": publisher})()

    def start(self):
        self.ready.emit("YOLO：測試 · MediaPipe：測試")

    def stop(self):
        self.finished.emit()

    def update(self, **changes):
        self.updates.update(changes)

    def take_output(self):
        out, self.output = self.output, None
        return out


class SceneTest(unittest.TestCase):
    def test_gl_axes(self):
        np.testing.assert_allclose(to_gl([[1, 2, 3]]), [[1, 3, -2]])

    def test_scene_contents(self):
        scene = build_scene(fake_output().people, point_cloud=True)
        self.assertEqual(len(scene.lines) % 2, 0)
        self.assertEqual(len(scene.lines), len(scene.line_colors))
        self.assertEqual(len(scene.joints), 2 * len(MEDIAPIPE33.body) + 1)  # 兩副骨架的身體關節 + 一個只有遮罩的人
        self.assertEqual(len(scene.cloud), 66)
        self.assertEqual([tid for tid, _, _ in scene.labels], [1, 2, 9])


class MainWindowTest(unittest.TestCase):
    def make(self, enable_3d=False):
        window = MainWindow(CFG, worker_factory=FakeWorker, enable_3d=enable_3d, camera_discover=None)
        window.show()
        return window

    def test_start_shows_frame_people_and_metrics(self):
        w = self.make()
        w.start()
        w.refresh()
        self.assertIsNotNone(w.last_image)
        self.assertEqual(sorted(w.people.cards), [1, 2, 9])
        self.assertIn("僅遮罩", w.people.cards[9].note.text())
        self.assertEqual(w.inspector.metrics["people"].text(), "3")
        self.assertIn("29.5", w.chips["fps"].text())
        self.assertEqual(w.inspector.start.text(), "停止")
        w.stop()
        self.assertIsNone(w.worker)
        self.assertEqual(w.inspector.start.text(), "開始")
        w.close()

    def test_pipeline_settings_reach_worker_and_display_settings_do_not(self):
        w = self.make()
        w.start()
        w.inspector.tracking_box.setChecked(False)
        w.inspector.confidence_slider.setValue(1000)
        w.inspector.layer_mask_box.setChecked(False)
        self.assertEqual(w.worker.updates["tracking"], False)
        self.assertAlmostEqual(w.worker.updates["confidence"], 0.9)
        self.assertNotIn("layer_mask", w.worker.updates)
        self.assertFalse(w.display["layer_mask"])
        w.stop()
        w.close()

    def test_settings_before_start_go_into_worker_config(self):
        w = self.make()
        w.inspector.use_mask_box.setChecked(False)
        self.assertEqual(w.worker_config()["pose"]["backend"], "rtmpose")  # 預設
        w.inspector.model.setCurrentIndex(3)  # MediaPipe lite
        cfg = w.worker_config()
        self.assertFalse(cfg["fusion"]["use_mask"])
        self.assertEqual(cfg["pose"]["backend"], "mediapipe")
        self.assertTrue(cfg["pose"]["model"].endswith("pose_landmarker_lite.task"))
        self.assertTrue(CFG["fusion"]["use_mask"])  # 原設定不被修改
        w.close()

    def test_recording_toggle(self):
        w = self.make()
        w.toggle_recording()  # 未開始時不能錄
        self.assertFalse(w.record_button.isChecked())
        w.start()
        w.toggle_recording()
        self.assertEqual(w.worker.updates["record"], {"with_images": True})
        w.on_recording("recordings/test", True)
        self.assertEqual(w.record_button.text(), "■ 停止錄製")
        w.toggle_recording()
        self.assertIsNone(w.worker.updates["record"])
        w.on_recording("recordings/test", False)
        self.assertIn("錄製已儲存", w.status.text())
        w.stop()
        w.close()

    def test_camera_selection(self):
        from astra_studio.sensors.discovery import Camera
        astra = Camera("Astra Pro HD Camera", 0, 1200, "uid-astra")
        mac = Camera("MacBook Pro相機", 1, 1200, "uid-mac")
        w = self.make()
        w.on_cameras([mac, astra])
        self.assertEqual(w.inspector.selected_camera(), astra)  # 預設選 Astra
        cfg = w.worker_config()
        self.assertEqual(cfg["source"], {"kind": "astra", "index": 0, "name": astra.name})
        self.assertEqual(cfg["rgb"]["fx"], CFG["rgb"]["fx"])
        w.inspector.camera.setCurrentIndex(0)  # 選 MacBook 相機
        self.assertIn("估計", w.inspector.source.text())
        cfg = w.worker_config()
        self.assertEqual(cfg["source"]["kind"], "rgb")
        self.assertAlmostEqual(cfg["rgb"]["fx"], 1280 / (2 * np.tan(np.radians(CFG["webcam"]["hfov_deg"] / 2))), places=3)
        self.assertEqual(CFG["rgb"]["fx"], 983.0)  # 原設定不被修改
        w.close()

    def test_start_waits_for_camera_discovery(self):
        """回歸：--start 在攝影機列舉完成前觸發時，曾直接用預設的 Astra 開啟，而不是使用者選的攝影機。"""
        import time
        from astra_studio.sensors.discovery import Camera
        cams = [Camera("Astra Pro HD Camera", 0, 1200, "a"), Camera("USB Webcam", 2, 1200, "b")]

        def slow_discover():
            time.sleep(0.3)
            return cams
        w = MainWindow(CFG, worker_factory=FakeWorker, enable_3d=False, camera_discover=slow_discover, initial_camera="2")
        w.refresh_cameras()
        w.start()  # 列舉還沒完成
        self.assertIsNone(w.worker)
        deadline = time.monotonic() + 5
        while w.worker is None and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(0.01)
        self.assertIsNotNone(w.worker)
        self.assertEqual(w.worker.cfg["source"]["kind"], "rgb")
        self.assertEqual(w.worker.cfg["source"]["index"], 2)
        w.stop()
        w.close()

    def test_stream_toggle(self):
        import copy
        cfg = copy.deepcopy(CFG)
        cfg["stream"]["port"] = 0  # 隨機可用埠
        w = MainWindow(cfg, worker_factory=FakeWorker, enable_3d=False, camera_discover=None)
        w.inspector.stream_enabled_box.setChecked(True)
        self.assertIsNotNone(w.stream)
        self.assertIn("ws://127.0.0.1:", w.inspector.stream_info.text())
        w.start()
        self.assertIs(w.worker.runner.publisher, w.stream)  # 開始時把串流交給擷取迴圈
        w.inspector.stream_enabled_box.setChecked(False)
        self.assertIsNone(w.stream)
        self.assertIsNone(w.worker.runner.publisher)
        self.assertEqual(w.inspector.stream_info.text(), "未啟用")
        w.stop()
        w.close()

    def test_initial_camera_and_depth_chip(self):
        from astra_studio.sensors.discovery import Camera
        cams = [Camera("Astra Pro HD Camera", 0, 1200, "a"), Camera("USB Webcam", 2, 1200, "b")]
        w = MainWindow(CFG, worker_factory=FakeWorker, enable_3d=False, camera_discover=None, initial_camera="2")
        w.on_cameras(cams)
        self.assertEqual(w.inspector.selected_camera().name, "USB Webcam")
        w.set_chips(True, False, 30.0, has_depth=False)
        self.assertIn("無深度", w.chips["depth"].text())
        w.close()

    def test_modes_render(self):
        w = self.make()
        w.start()
        out, _ = w.worker.output
        for mode in ("overlay", "depth", "skeleton", "contour", "split"):
            w.set_mode(mode)
            img = w.compose(out)
            self.assertEqual(img.shape, (720, 1280, 3))
        w.set_mode("skeleton")
        self.assertEqual(int(w.compose(out)[0, 0].sum()), 0)  # 黑底
        w.select_mode("contour")
        self.assertTrue(w.mode_buttons["contour"].isChecked())
        self.assertFalse(w.mode_buttons["overlay"].isChecked())
        img = w.compose(out)
        self.assertEqual(int(img[400, 375].sum()), 0)   # 遮罩內部不填色
        self.assertGreater(int(img[400, 299:302].sum()), 0)  # 遮罩左邊界有輪廓線
        w.stop()
        w.close()

    def test_view3d_widget(self):
        try:
            w = self.make(enable_3d=True)
        except Exception as error:  # offscreen 平台可能沒有 OpenGL
            self.skipTest(f"無 OpenGL：{error}")
        w.set_mode("split")
        w.start()
        w.refresh()
        self.assertEqual(set(w.view3d._labels), {1, 2, 9})
        w.stop()
        w.close()


if __name__ == "__main__":
    unittest.main()
