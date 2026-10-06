"""perception、融合管線測試：用真實模型跑範例影像（tests/data/person.jpg，MediaPipe 官方範例圖）。

範例圖或模型不存在時自動略過（由 scripts/download_models.sh 下載）；不需要相機。
"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from astra_studio.config import load_config

CFG = load_config()
IMAGE = Path(__file__).parent / "data" / "person.jpg"
if not IMAGE.exists():
    raise unittest.SkipTest("缺 tests/data/person.jpg，請先執行 scripts/download_models.sh")


def frame():
    return cv2.resize(cv2.imread(str(IMAGE)), (1280, 720))


@unittest.skipUnless(Path(CFG["segmentation"]["model"]).exists(), "缺 YOLO 模型")
class SegmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from astra_studio.perception.segmentation import PersonSegmenter
        cls.seg = PersonSegmenter(CFG["segmentation"])

    def test_finds_person_with_full_resolution_mask(self):
        img = frame()
        result = self.seg(img, track=False)
        self.assertGreaterEqual(len(result.people), 1)
        p = result.people[0]
        self.assertEqual(p.mask.shape, img.shape[:2])
        self.assertIsNone(p.track_id)
        # 遮罩應幾乎都落在框內（檢查沒有 letterbox 偏移）
        x1, y1, x2, y2 = p.box.astype(int)
        inside = p.mask[y1:y2 + 1, x1:x2 + 1].sum() / p.mask.sum()
        self.assertGreater(inside, 0.97)

    def test_tracking_assigns_stable_id(self):
        self.seg.reset()
        img = frame()
        ids = [tuple(p.track_id for p in self.seg(img, track=True).people) for _ in range(3)]
        self.assertTrue(all(i is not None for i in ids[-1]))
        self.assertEqual(ids[1], ids[2])


@unittest.skipUnless(Path(CFG["pose"]["rtm_model"]).exists() and Path(CFG["segmentation"]["model"]).exists(), "缺 RTMPose / YOLO 模型")
class RTMPoseTest(unittest.TestCase):
    def test_top_down_on_yolo_box(self):
        from astra_studio.core.skeleton_format import HALPE26
        from astra_studio.perception.pose_rtm import RTMPoseEstimator
        from astra_studio.perception.segmentation import PersonSegmenter
        img = frame()
        box = PersonSegmenter(CFG["segmentation"])(img, track=False).people[0].box
        est = RTMPoseEstimator(CFG["pose"])
        poses = est(img, [box, box * 0.5])
        self.assertEqual(len(poses), 2)
        p = poses[0]
        self.assertEqual(p.fmt, HALPE26)
        self.assertEqual(p.pixels.shape, (26, 2))
        self.assertIsNone(p.world)
        self.assertGreater(float(np.median(p.visibility)), 0.5)
        x1, y1, x2, y2 = box
        pad = 0.15 * max(x2 - x1, y2 - y1)
        inside = (p.pixels[:, 0] > x1 - pad) & (p.pixels[:, 0] < x2 + pad) & (p.pixels[:, 1] > y1 - pad) & (p.pixels[:, 1] < y2 + pad)
        self.assertTrue(inside.all())
        est.max_people = 1
        self.assertIsNone(est(img, [box * 0.5, box])[0])  # 超過上限時留下較大的框


@unittest.skipUnless(Path(CFG["pose"]["model"]).exists(), "缺 MediaPipe 模型")
class PoseTest(unittest.TestCase):
    def test_detects_one_person(self):
        from astra_studio.perception.pose import PoseEstimator
        est = PoseEstimator(CFG["pose"])
        try:
            poses = est(frame())
        finally:
            est.close()
        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0].pixels.shape, (33, 2))


@unittest.skipUnless(Path(CFG["segmentation"]["model"]).exists() and Path(CFG["pose"]["model"]).exists(), "缺模型")
class PipelineTest(unittest.TestCase):
    """真實模型 + 合成深度（整面 2.2 m）跑完整融合流程。"""

    def test_fused_person_has_mask_skeleton_and_position(self):
        for backend in ("rtmpose", "mediapipe"):
            with self.subTest(backend=backend):
                self._check_backend(backend)

    def _check_backend(self, backend):
        import copy
        from astra_studio.pipeline.pipeline import Pipeline
        cfg = copy.deepcopy(CFG)
        cfg["pose"]["backend"] = backend
        pipe = Pipeline(cfg)
        self.assertEqual(pipe.top_down, backend == "rtmpose")
        try:
            img = frame()
            depth = np.full((480, 640), 2200, np.uint16)
            outs = [pipe.process(img, depth, i / 30) for i in range(4)]
        finally:
            pipe.close()
        people = outs[-1].people
        self.assertEqual(len(people), 1)
        p = people[0]
        self.assertIsNotNone(p.segment)
        self.assertIsNotNone(p.skeleton)
        self.assertEqual(p.track_id, outs[1].people[0].track_id)  # BoT-SORT ID 穩定
        self.assertGreater(p.measured_ratio, 0.6)
        self.assertAlmostEqual(float(p.centroid[2]), 2.2, delta=0.05)
        for key in ("segmentation", "pose", "fusion", "total"):
            self.assertIn(key, outs[-1].timings)

    def test_gpu_memory_does_not_grow(self):
        """回歸測試：mediapipe GPU delegate 曾每幀洩漏約 14 MB Metal 記憶體，30 fps 下一分鐘內耗盡。"""
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest("無 MPS")
        from astra_studio.pipeline.pipeline import Pipeline
        pipe = Pipeline(CFG)
        try:
            img, depth = frame(), np.full((480, 640), 2200, np.uint16)
            for i in range(15):
                pipe.process(img, depth, i / 30)
            start = torch.mps.driver_allocated_memory()
            for i in range(90):
                pipe.process(img, depth, (15 + i) / 30)
            growth_mb = (torch.mps.driver_allocated_memory() - start) / 2**20
        finally:
            pipe.close()
        self.assertLess(growth_mb, 100, f"90 幀內 Metal 記憶體成長 {growth_mb:.0f} MB")


if __name__ == "__main__":
    unittest.main()
