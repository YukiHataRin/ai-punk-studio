"""core 層單元測試：不需要相機與模型。"""

import unittest
from dataclasses import replace

import numpy as np

from astra_studio.config import load_config
from astra_studio.core.filters import OneEuroFilter
from astra_studio.core.geometry import Intrinsics, rotation_matrix
from astra_studio.core.lift import lift_person
from astra_studio.core.registration import DepthToRgb
from astra_studio.core.fusion import associate, depth_mask, mask_centroid
from astra_studio.core.tracker import PoseTracker, SkeletonSmoother
from astra_studio.core.types import PoseObservation, SegmentedPerson, TrackedPerson
from astra_studio.render.colors import track_bgr
from astra_studio.render.overlay import draw_masks

CFG = load_config()
RGB_K = Intrinsics.from_config(CFG["rgb"])


def standing_person(x=640, y=360, scale=1.0):
    """在 (x, y) 附近的一個假人：33 個關節排成一個人形，world 座標與像素一致。"""
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-1, 1, (33, 2)) * [60, 150]
    offsets[[11, 12, 23, 24]] = [[-40, -80], [40, -80], [-30, 40], [30, 40]]  # 肩、髖
    pixels = (np.array([x, y]) + offsets * scale).astype(np.float32)
    world = np.c_[offsets / 300, np.zeros(33)].astype(np.float32)
    return PoseObservation(pixels, np.ones(33, np.float32), world)


class GeometryTest(unittest.TestCase):
    def test_backproject_center_is_on_axis(self):
        p = RGB_K.backproject(RGB_K.cx, RGB_K.cy, 2.0)
        np.testing.assert_allclose(p, [0, 0, 2.0], atol=1e-9)

    def test_backproject_vectorized(self):
        p = RGB_K.backproject(np.array([0.0, 1280.0]), np.array([0.0, 720.0]), np.array([1.0, 3.0]))
        self.assertEqual(p.shape, (2, 3))
        np.testing.assert_allclose(p[:, 2], [1.0, 3.0])

    def test_rotation_is_orthonormal(self):
        R = rotation_matrix([10, -20, 30])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


class RegistrationTest(unittest.TestCase):
    def test_flat_plane_keeps_depth(self):
        reg = DepthToRgb(CFG)
        out = reg(np.full((480, 640), 2000, np.uint16))
        self.assertEqual(out.shape, (360, 640))
        valid = out[out > 0]
        self.assertGreater(valid.size, 0.2 * out.size)
        np.testing.assert_allclose(np.median(valid), 2000, atol=2)

    def test_out_of_range_is_dropped(self):
        reg = DepthToRgb(CFG)
        out = reg(np.full((480, 640), CFG["depth"]["max_mm"] + 100, np.uint16))
        self.assertEqual(int(out.max()), 0)


class LiftTest(unittest.TestCase):
    def test_flat_depth_measures_every_joint(self):
        person = standing_person()
        depth = np.full((360, 640), 2000, np.uint16)
        sk = lift_person(person, depth, 0.5, RGB_K, CFG["skeleton"])
        self.assertTrue(sk.measured.all())
        np.testing.assert_allclose(sk.points[:, 2], 2.0 + CFG["skeleton"]["joint_depth_offset"], atol=1e-3)

    def test_background_pixels_are_gated_out(self):
        person = standing_person()
        rng = np.random.default_rng(1)
        depth = np.full((360, 640), 2000, np.uint16)
        depth[rng.random(depth.shape) < 0.3] = 4500  # 30% 背景
        sk = lift_person(person, depth, 0.5, RGB_K, CFG["skeleton"])
        self.assertLess(float(sk.points[:, 2].max()), 2.2)

    def test_no_depth_falls_back_to_world_landmarks(self):
        sk = lift_person(standing_person(), np.zeros((360, 640), np.uint16), 0.5, RGB_K, CFG["skeleton"])
        self.assertFalse(sk.measured.any())
        self.assertGreater(sk.distance, 2.0)


class FilterTrackerTest(unittest.TestCase):
    def test_one_euro_reduces_noise(self):
        rng = np.random.default_rng(2)
        f = OneEuroFilter(0.8, 0.6)
        noisy = rng.normal(0, 0.05, 90)
        smooth = np.array([f(v, i / 30) for i, v in enumerate(noisy)])
        self.assertLess(smooth[30:].std(), noisy[30:].std() / 2)

    def test_tracker_keeps_id_and_creates_new_ones(self):
        tr = PoseTracker(CFG["smoothing"], 1280)
        ids0 = tr.assign([standing_person(400), standing_person(900)], 0.0)
        ids1 = tr.assign([standing_person(900), standing_person(410)], 1 / 30)
        self.assertEqual(ids1, ids0[::-1])
        # 消失超過 timeout 後再出現 → 新 ID
        later = tr.assign([standing_person(400)], 1.0)
        self.assertNotIn(later[0], ids0)

    def test_smoother_is_keyed_by_track_id(self):
        sm = SkeletonSmoother(CFG["smoothing"])
        depth = np.full((360, 640), 2000, np.uint16)
        pose = standing_person()
        sk = lift_person(pose, depth, 0.5, RGB_K, CFG["skeleton"])
        sm.smooth(TrackedPerson(1, pose=pose, skeleton=sk), 0.0)
        jumped = replace(sk, points=sk.points + 1.0)
        out1 = sm.smooth(TrackedPerson(1, pose=pose, skeleton=jumped), 1 / 30)
        out2 = sm.smooth(TrackedPerson(2, pose=pose, skeleton=jumped), 1 / 30)
        # 同一 ID 被平滑（不會立刻跳 1 m）；新 ID 直接採用輸入
        self.assertLess(float((out1.skeleton.points - sk.points).mean()), 0.9)
        np.testing.assert_allclose(out2.skeleton.points, jumped.points, atol=1e-5)
        sm.prune(10.0)
        self.assertEqual(len(sm._filters), 0)


def rect_mask(x1, y1, x2, y2, shape=(720, 1280)):
    m = np.zeros(shape, bool)
    m[y1:y2, x1:x2] = True
    return m


class FusionTest(unittest.TestCase):
    def test_associate_matches_pose_to_its_own_mask(self):
        a, b = standing_person(400), standing_person(900)
        seg_a = SegmentedPerson(rect_mask(300, 150, 500, 600), np.array([300, 150, 500, 600]), 0.9, 7)
        seg_b = SegmentedPerson(rect_mask(800, 150, 1000, 600), np.array([800, 150, 1000, 600]), 0.9, 3)
        self.assertEqual(associate([a, b], [seg_b, seg_a]), {0: 1, 1: 0})

    def test_unrelated_mask_is_not_matched(self):
        seg = SegmentedPerson(rect_mask(0, 0, 100, 100), np.array([0, 0, 100, 100]), 0.9, 1)
        self.assertEqual(associate([standing_person(900)], [seg]), {})

    def test_mask_rejects_nearer_occluder_inside_gate(self):
        """另一個人站在前方 0.4 m（仍在 0.6 m 門檻內）並遮住窗口一半：只有遮罩能排除他。"""
        pose = standing_person(640, 360)
        depth = np.full((360, 640), 2000, np.uint16)
        mask_full = rect_mask(0, 0, 1280, 720)
        occluder_cols = slice(0, 320)  # 深度圖左半邊是前方的人
        depth[:, occluder_cols] = 1600
        mask_full[:, :640] = False      # 他不屬於這個人的遮罩（原圖解析度左半）
        segment = SegmentedPerson(mask_full, np.zeros(4), 0.9, 1)
        small = depth_mask(segment, depth.shape, erode_px=0)
        with_mask = lift_person(pose, depth, 0.5, RGB_K, CFG["skeleton"], small)
        without = lift_person(pose, depth, 0.5, RGB_K, CFG["skeleton"])
        offset = CFG["skeleton"]["joint_depth_offset"]
        z_mask = with_mask.points[with_mask.measured, 2]
        np.testing.assert_allclose(z_mask, 2.0 + offset, atol=1e-3)
        self.assertLess(float(without.points[without.measured, 2].min()), 1.9)

    def test_distance_text_marks_estimates(self):
        pose = standing_person()
        measured = lift_person(pose, np.full((360, 640), 2000, np.uint16), 0.5, RGB_K, CFG["skeleton"])
        estimated = lift_person(pose, np.zeros((360, 640), np.uint16), 0.5, RGB_K, CFG["skeleton"])
        self.assertFalse(TrackedPerson(1, pose=pose, skeleton=measured).distance_text().startswith("≈"))
        self.assertTrue(TrackedPerson(1, pose=pose, skeleton=estimated).distance_text().startswith("≈"))
        self.assertEqual(TrackedPerson(1).distance_text(), "")

    def test_mask_centroid(self):
        depth = np.full((360, 640), 2500, np.uint16)
        segment = SegmentedPerson(rect_mask(600, 300, 680, 420), np.zeros(4), 0.9, 1)
        small = depth_mask(segment, depth.shape)
        c = mask_centroid(small, depth, Intrinsics.from_config(CFG["rgb"], 0.5))
        self.assertAlmostEqual(float(c[2]), 2.5, places=3)
        self.assertLess(abs(float(c[0])), 0.01)
        self.assertIsNone(mask_centroid(small, np.zeros_like(depth), RGB_K))


class OverlayTest(unittest.TestCase):
    def test_mask_colors(self):
        img = np.full((4, 4, 3), 100, np.uint8)
        mask = np.zeros((4, 4), bool)
        mask[0, 0] = True
        person = SegmentedPerson(mask, np.zeros(4), 0.9, track_id=2)
        only = draw_masks(img, [person], mask_only=True)
        self.assertEqual(tuple(only[0, 0]), track_bgr(2))
        self.assertEqual(int(only[1, 1].sum()), 0)
        blended = draw_masks(img, [person], opacity=0.5)
        self.assertEqual(tuple(blended[1, 1]), (100, 100, 100))


if __name__ == "__main__":
    unittest.main()
