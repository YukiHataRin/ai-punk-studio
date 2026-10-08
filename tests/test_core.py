"""core 層單元測試：不需要相機與模型。"""

import unittest
from dataclasses import replace

import numpy as np

from aipunk_studio.config import load_config
from aipunk_studio.core.filters import OneEuroFilter
from aipunk_studio.core.geometry import Intrinsics, rotation_matrix
from aipunk_studio.core.skeleton_format import FORMATS, HALPE26, MEDIAPIPE33
from aipunk_studio.core.lift import estimate_distance, lift_person
from aipunk_studio.core.registration import DepthToRgb
from aipunk_studio.core.fusion import associate, depth_mask, mask_centroid, match_previous_boxes
from aipunk_studio.core.tracker import PoseTracker, SkeletonSmoother
from aipunk_studio.core.types import PoseObservation, SegmentedPerson, TrackedPerson
from aipunk_studio.render.colors import track_bgr
from aipunk_studio.render.overlay import draw_masks

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


class TelemetryTest(unittest.TestCase):
    def test_onnxruntime_telemetry_disabled_before_import(self):
        """回歸：onnxruntime 遙測的上傳執行緒曾讓程式結束時 abort（exit 134），必須在匯入前關閉。"""
        import subprocess
        import sys
        code = "import os, aipunk_studio, onnxruntime; print(os.environ.get('ORT_DISABLE_TELEMETRY'))"
        env = {k: v for k, v in __import__("os").environ.items() if k != "ORT_DISABLE_TELEMETRY"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(out.stdout.strip(), "1")


class SkeletonFormatTest(unittest.TestCase):
    def test_formats_are_consistent(self):
        for fmt in FORMATS.values():
            idx = [i for c in fmt.connections for i in c] + [*fmt.shoulders, *fmt.hips, *fmt.ankles, fmt.head, *fmt.face]
            self.assertTrue(all(0 <= i < fmt.size for i in idx), fmt.name)
            self.assertEqual(len(set(fmt.names)), fmt.size, fmt.name)
            self.assertNotIn(fmt.head, fmt.face)
        self.assertEqual(HALPE26.size, 26)
        self.assertEqual(HALPE26.names[HALPE26.hips[0]], "left_hip")
        self.assertEqual(FORMATS["mediapipe33"].names[23], "left_hip")


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

    def test_rtmpose_fallback_puts_missing_joints_on_body_depth_plane(self):
        """沒有 world landmarks（RTMPose）時，量不到的關節沿用 2D 位置、深度取實測關節中位數。"""
        rng = np.random.default_rng(3)
        pixels = (np.array([640, 360]) + rng.uniform(-60, 60, (26, 2))).astype(np.float32)
        pose = PoseObservation(pixels, np.ones(26, np.float32), None, HALPE26)
        depth = np.full((360, 640), 2000, np.uint16)
        u, v = (pixels[9] * 0.5).astype(int)  # 左手腕附近挖一個沒有深度的洞
        depth[v - 6:v + 7, u - 6:u + 7] = 0
        sk = lift_person(pose, depth, 0.5, RGB_K, CFG["skeleton"])
        self.assertFalse(sk.measured[9])
        self.assertGreater(sk.measured.sum(), 20)
        z = 2.0 + CFG["skeleton"]["joint_depth_offset"]
        np.testing.assert_allclose(sk.points[9], RGB_K.backproject(*pixels[9], z), atol=1e-4)
        self.assertEqual(sk.fmt, HALPE26)

    def test_few_outlier_joints_do_not_move_the_body_plane(self):
        """只有頭部一點量到深度且是背景（遮罩邊緣）時，其餘關節要放在遮罩的深度上，不能跟著那個錯值。"""
        pixels = np.array([  # Halpe26 站姿：臉、肩肘腕、髖膝踝、頭頂、頸、骨盆、腳趾、腳跟
            [640, 150], [630, 140], [650, 140], [620, 145], [660, 145], [600, 220], [680, 220],
            [580, 300], [700, 300], [570, 380], [710, 380], [615, 400], [665, 400], [612, 520], [668, 520],
            [610, 640], [670, 640], [640, 110], [640, 200], [640, 400], [600, 700], [680, 700],
            [590, 705], [690, 705], [615, 690], [665, 690]], np.float32)
        pose = PoseObservation(pixels, np.ones(26, np.float32), None, HALPE26)
        depth = np.full((360, 640), 700, np.uint16)  # 人在 0.7 m
        win = CFG["skeleton"]["depth_window"]
        uv = (pixels * 0.5).astype(int)
        for u, v in uv:  # 每個關節附近都量不到深度（太近、反光等）
            depth[v - 3 * win:v + 3 * win + 1, u - 3 * win:u + 3 * win + 1] = 0
        u, v = uv[0]
        depth[v - win:v + win + 1, u - win:u + win + 1] = 1100  # 鼻子量到後方背景
        mask = np.ones(depth.shape, bool)
        sk = lift_person(pose, depth, 0.5, RGB_K, CFG["skeleton"], mask)
        self.assertTrue(sk.measured.any())
        self.assertFalse(sk.measured[list(HALPE26.torso)].any())  # 只有臉部附近量到（錯的 1.1 m）
        self.assertAlmostEqual(float(np.median(sk.points[~sk.measured, 2])), 0.7, delta=0.02)

    def test_few_outlier_joints_do_not_move_mediapipe_skeleton(self):
        """MediaPipe（有 world landmarks）同樣不能被少數臉部錯值帶走：只有左耳量到背景時，骨架要放在遮罩深度上。"""
        from aipunk_studio.core.lift import body_mask
        pose = standing_person()
        depth = np.full((360, 640), 700, np.uint16)
        win = CFG["skeleton"]["depth_window"]
        uv = (pose.pixels * 0.5).astype(int)
        for u, v in uv:
            depth[v - 3 * win:v + 3 * win + 1, u - 3 * win:u + 3 * win + 1] = 0
        u, v = uv[7]  # 左耳
        depth[v - win:v + win + 1, u - win:u + win + 1] = 1100
        sk = lift_person(pose, depth, 0.5, RGB_K, CFG["skeleton"], np.ones(depth.shape, bool))
        self.assertTrue(sk.measured[7])
        self.assertLess(int((sk.measured & body_mask(MEDIAPIPE33)).sum()), 3)
        self.assertAlmostEqual(float(np.median(sk.points[~sk.measured, 2])), 0.7, delta=0.02)

    def test_no_depth_estimates_distance_from_torso_length(self):
        """沒有深度時用軀幹長估距離：軀幹像素長 = fy × 0.5 m ÷ 距離。"""
        z_true = 3.0
        torso_px = RGB_K.fy * CFG["skeleton"]["est_torso_m"] / z_true
        pixels = np.tile([640.0, 360.0], (26, 1)).astype(np.float32)
        pixels[[5, 6], 1] = 300                      # 肩
        pixels[[11, 12], 1] = 300 + torso_px          # 髖
        pixels[[5, 11], 0], pixels[[6, 12], 0] = 600, 680
        pose = PoseObservation(pixels, np.ones(26, np.float32), None, HALPE26)
        self.assertAlmostEqual(estimate_distance(pose, RGB_K, CFG["skeleton"]), z_true, places=3)
        sk = lift_person(pose, np.zeros((360, 640), np.uint16), 0.5, RGB_K, CFG["skeleton"])
        self.assertFalse(sk.measured.any())
        np.testing.assert_allclose(sk.points[:, 2], z_true, atol=1e-3)  # RTMPose：落在估計距離的平面上
        hidden = PoseObservation(pixels, np.zeros(26, np.float32), None, HALPE26)
        self.assertIsNone(estimate_distance(hidden, RGB_K, CFG["skeleton"]))

    def test_fov_intrinsics(self):
        from aipunk_studio.sensors.webcam import intrinsics_from_fov
        k = intrinsics_from_fov(1280, 720, 90.0)
        self.assertAlmostEqual(k["fx"], 640.0)
        self.assertEqual((k["cx"], k["cy"]), (640.0, 360.0))

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

    def test_match_previous_boxes(self):
        def seg(box, tid):
            return SegmentedPerson(np.zeros((2, 2), bool), np.array(box, float), 0.9, tid)
        now = [seg([500, 100, 600, 400], 7), seg([100, 100, 200, 400], 3), seg([900, 100, 1000, 400], None)]
        prev = [(3, np.array([105, 98, 205, 402.])),      # 同 ID → 對到 index 1
                (None, np.array([905, 100, 1005, 400.])),  # 沒 ID，靠 IoU → index 2
                (8, np.array([0, 600, 50, 700.]))]         # 已離開畫面 → 不配
        self.assertEqual(match_previous_boxes(prev, now), {0: 1, 1: 2})

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
