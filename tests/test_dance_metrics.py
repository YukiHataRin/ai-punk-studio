"""舞蹈指標測試：用解析解已知的合成動作驗證九項指標（不需要相機與模型）。"""

import unittest

import numpy as np

from aipunk_studio.core import dance_metrics as dm
from aipunk_studio.core.skeleton_format import HALPE26, MEDIAPIPE33
from aipunk_studio.core.types import PoseObservation, Skeleton3D, TrackedPerson

FPS = 30.0


def standing():
    """H36M 站姿（公尺，y 向下，原點約在骨盆）。"""
    p = np.zeros((17, 3))
    p[dm.PELVIS] = [0, 0, 2]
    p[dm.R_HIP], p[dm.L_HIP] = [-0.1, 0, 2], [0.1, 0, 2]
    p[dm.R_KNEE], p[dm.L_KNEE] = [-0.1, 0.45, 2], [0.1, 0.45, 2]
    p[dm.R_ANKLE], p[dm.L_ANKLE] = [-0.1, 0.9, 2], [0.1, 0.9, 2]
    p[dm.SPINE], p[dm.THORAX], p[dm.NECK], p[dm.HEAD] = [0, -0.25, 2], [0, -0.38, 2], [0, -0.5, 2], [0, -0.65, 2]
    p[dm.R_SHOULDER], p[dm.L_SHOULDER] = [-0.2, -0.5, 2], [0.2, -0.5, 2]
    p[dm.R_ELBOW], p[dm.L_ELBOW] = [-0.2, -0.2, 2], [0.2, -0.2, 2]
    p[dm.R_WRIST], p[dm.L_WRIST] = [-0.2, 0.05, 2], [0.2, 0.05, 2]
    return p


def run(engine, frames, floor_y=None):
    out = None
    for i, pos in enumerate(frames):
        out = engine.update(pos, i / FPS, floor_y)
    return out


def rotate_forearm(p, side_elbow, side_wrist, angle):
    """前臂在 x–y 平面繞手肘轉到 angle（rad）。"""
    q = p.copy()
    length = np.linalg.norm(p[side_wrist] - p[side_elbow])
    q[side_wrist] = p[side_elbow] + length * np.array([np.sin(angle), np.cos(angle), 0])
    return q


class EngineTest(unittest.TestCase):
    def test_still_body_has_no_motion(self):
        m = run(dm.DanceMetricsEngine(), [standing()] * 10)
        self.assertAlmostEqual(m["energy"], 0.0)
        self.assertAlmostEqual(m["torque"], 0.0)
        self.assertAlmostEqual(m["jerk"], 0.0)
        self.assertAlmostEqual(m["sync_velocity"], 1.0, places=3)  # 兩側都不動 = 平衡

    def test_intensity_of_rotating_forearm(self):
        omega = np.pi  # rad/s
        frames = [rotate_forearm(standing(), dm.L_ELBOW, dm.L_WRIST, omega * i / FPS) for i in range(10)]
        m = run(dm.DanceMetricsEngine(), frames)
        expected = (dm.LIMB_WEIGHTS["L_Arm"] / 2) * omega ** 2  # 只有左前臂在轉
        self.assertAlmostEqual(m["energy"], expected, places=3)
        # 只有左邊動：平衡 = 1 − ωL / (ωL + 靜止門檻 × 2 組肢體)，ωL 為左臂兩段的平均角速度
        omega_l = omega / 2
        self.assertAlmostEqual(m["sync_velocity"], 1 - omega_l / (omega_l + 2 * dm.OMEGA_REST), places=3)
        fast = [rotate_forearm(standing(), dm.L_ELBOW, dm.L_WRIST, 6 * np.pi * i / FPS) for i in range(10)]
        self.assertLess(run(dm.DanceMetricsEngine(), fast)["sync_velocity"], 0.15)  # 動作明顯時接近原公式的 0

    def test_mirrored_limbs_are_synchronized(self):
        def frames(phase, amplitude=1.0):
            out = []
            for i in range(40):
                speed = 1 + np.sin(i / 3)  # 速度隨時間變化，相關係數才有意義
                ang_l = amplitude * np.sin(i / 5)
                ang_r = amplitude * np.sin(i / 5 + phase)
                p = rotate_forearm(standing(), dm.L_ELBOW, dm.L_WRIST, ang_l * speed)
                out.append(rotate_forearm(p, dm.R_ELBOW, dm.R_WRIST, -ang_r * speed))
            return out
        same = run(dm.DanceMetricsEngine(), frames(0.0))
        self.assertGreater(same["sync_correlation"], 0.9)
        self.assertGreater(same["sync_velocity"], 0.95)
        tiny = run(dm.DanceMetricsEngine(), frames(0.0, amplitude=0.1))  # 幅度接近抖動：刻意縮向 0
        self.assertLess(tiny["sync_correlation"], same["sync_correlation"])
        self.assertGreater(tiny["sync_correlation"], 0.0)

    def test_still_jitter_does_not_swing_symmetry(self):
        """靜止但骨架有 1.5 mm 抖動：平衡應穩定接近 1、協調接近 0（原公式會在 0–1、−1–1 之間亂跳）。"""
        rng = np.random.default_rng(7)
        frames = [standing() + rng.normal(0, 0.0015, (17, 3)) for _ in range(90)]
        e = dm.DanceMetricsEngine()
        out = [e.update(p, i / FPS) for i, p in enumerate(frames)][30:]
        balance = np.array([m["sync_velocity"] for m in out])
        corr = np.array([m["sync_correlation"] for m in out])
        self.assertGreater(np.median(balance), 0.8)
        self.assertLess(balance.std(), 0.1)
        self.assertLess(np.abs(corr).max(), 0.4)
        wl, wr = np.array(e.omega_l), np.array(e.omega_r)  # 同一段資料套原公式，作為對照
        original = 1 - np.abs(wl - wr) / (np.maximum(wl, wr) + 1e-6)
        self.assertGreater(original.std(), 1.5 * balance.std())

    def test_undetected_joints_are_ignored(self):
        """腳沒入鏡（骨架模型猜的腳在亂跳）、上半身靜止：猜的腳不能被當成動作。"""
        rng = np.random.default_rng(3)
        detected = np.ones(17, bool)
        detected[[dm.L_KNEE, dm.R_KNEE, dm.L_ANKLE, dm.R_ANKLE]] = False
        e = dm.DanceMetricsEngine()
        for i in range(10):
            p = standing()
            p[[dm.L_KNEE, dm.R_KNEE, dm.L_ANKLE, dm.R_ANKLE]] += rng.normal(0, 0.1, (4, 3))
            m = e.update(p, i / FPS, floor_y=0.93, detected=detected)
        self.assertAlmostEqual(m["energy"], 0.0)
        self.assertAlmostEqual(m["torque"], 0.0)
        self.assertAlmostEqual(m["jerk"], 0.0)
        self.assertAlmostEqual(m["sync_velocity"], 1.0)
        self.assertAlmostEqual(m["curvature"], 0.0)  # 只平均偵測到的兩個手腕
        self.assertIsNone(m["height"])  # 看不到腳：重心高度與晃動不可靠
        self.assertIsNone(m["sway"])
        nothing = dm.DanceMetricsEngine()
        for i in range(3):
            m = nothing.update(standing(), i / FPS, detected=np.zeros(17, bool))
        self.assertTrue(all(v is None for v in m.values()))

    def test_expansion_unit_cube(self):
        p = np.full((17, 3), 0.5)  # 其餘關節在內部
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        p[:8] = corners
        all_seen = np.ones(17, bool)
        self.assertAlmostEqual(dm.DanceMetricsEngine._expansion(p, all_seen), 1.0, places=6)
        self.assertEqual(dm.DanceMetricsEngine._expansion(np.zeros((17, 3)), all_seen), 0.0)  # 退化不會出錯
        flat = np.random.default_rng(1).normal(size=(17, 3))
        flat[:, 2] = 2.0  # 沒有深度時所有關節在同一深度平面 → 扁平
        self.assertEqual(dm.DanceMetricsEngine._expansion(flat, all_seen), 0.0)
        half = all_seen.copy()
        half[[1, 3, 5, 7]] = False  # z = 1 的四個角沒偵測到 → 底面與中心點構成的金字塔
        self.assertAlmostEqual(dm.DanceMetricsEngine._expansion(p, half), 1 / 6, places=6)
        self.assertIsNone(dm.DanceMetricsEngine._expansion(p, np.arange(17) < 3))  # 少於 4 個關節

    def test_height_above_floor_and_sway(self):
        p = standing()
        com = dm.MASS_WEIGHTS @ p
        seen = np.ones(17, bool)
        height, sway = dm.DanceMetricsEngine._stability(p, seen, floor_y=0.93)
        self.assertAlmostEqual(height, 0.93 - com[1])
        self.assertAlmostEqual(sway, 0.0, places=6)  # 左右對稱，重心正好在兩腳中點上方
        leaning = p.copy()
        leaning[[dm.L_ANKLE, dm.R_ANKLE], 0] += 0.2  # 兩腳往右移 20 cm
        _, sway = dm.DanceMetricsEngine._stability(leaning, seen, floor_y=0.93)
        # 腳踝本身也有質量，移動腳會帶著重心移一點：0.2 × (1 − 兩腳踝質量權重)
        self.assertAlmostEqual(sway, 0.2 * (1 - 2 * dm.MASS_WEIGHTS[dm.L_ANKLE]), places=6)
        self.assertIsNone(dm.DanceMetricsEngine._stability(p, seen, floor_y=None)[0])
        no_feet = seen.copy()
        no_feet[dm.L_ANKLE] = False
        self.assertEqual(dm.DanceMetricsEngine._stability(p, no_feet, floor_y=0.93), (None, None))

    def test_curvature_of_circular_wrist(self):
        r, w = 0.5, 2 * np.pi * 0.5  # 半徑 0.5 m、每秒半圈 → 速度 1.57 m/s
        frames = []
        for i in range(10):
            p = standing()
            p[dm.L_WRIST] = [r * np.cos(w * i / FPS), r * np.sin(w * i / FPS), 2]
            frames.append(p)
        m = run(dm.DanceMetricsEngine(), frames)
        self.assertAlmostEqual(m["curvature"] * 4, 1 / r, delta=0.05)  # 四個末端平均，只有左手腕在動

    def test_metrics_need_history(self):
        e = dm.DanceMetricsEngine()
        self.assertTrue(all(v is None for v in e.update(standing(), 0.0).values()))
        second = e.update(standing(), 1 / FPS)
        self.assertIsNotNone(second["energy"])
        self.assertIsNone(second["torque"])  # 角加速度需要三幀


class MappingTest(unittest.TestCase):
    def test_both_formats_map_to_same_h36m(self):
        rng = np.random.default_rng(0)
        halpe = rng.normal(size=(26, 3))
        mp = np.zeros((33, 3))
        for i, name in enumerate(HALPE26.names):
            if name in MEDIAPIPE33.names:
                mp[MEDIAPIPE33.names.index(name)] = halpe[i]
        np.testing.assert_allclose(dm.to_h36m(halpe, HALPE26), dm.to_h36m(mp, MEDIAPIPE33))
        h = dm.to_h36m(halpe, HALPE26)
        np.testing.assert_allclose(h[dm.PELVIS], (halpe[11] + halpe[12]) / 2)
        np.testing.assert_allclose(h[dm.NECK], (halpe[5] + halpe[6]) / 2)
        np.testing.assert_allclose(h[dm.HEAD], halpe[0])


def person(tid, points, measured=True, visibility=None):
    pose = None if visibility is None else PoseObservation(np.zeros((26, 2), np.float32), visibility, None, HALPE26)
    return TrackedPerson(tid, pose=pose,
                         skeleton=Skeleton3D(points.astype(np.float32), np.full(26, measured), 2.0, HALPE26))


class MetricJointsTest(unittest.TestCase):
    def test_coverage_counts_only_the_13_metric_joints(self):
        measured = np.zeros(26, bool)
        measured[[17, 18, 19, 20, 21, 22, 23, 24, 25]] = True  # 頭頂、頸、骨盆、腳趾、腳跟：指標用不到
        sk = Skeleton3D(np.zeros((26, 3), np.float32), measured, 2.0, HALPE26)
        self.assertEqual(dm.metric_joint_coverage(sk), (0, 13))
        self.assertFalse(TrackedPerson(1, skeleton=sk).metrics_measured)
        measured[[9, 10]] = True  # 雙手腕
        self.assertEqual(dm.metric_joint_coverage(sk), (2, 13))
        self.assertTrue(TrackedPerson(1, skeleton=sk).metrics_measured)
        mp = Skeleton3D(np.zeros((33, 3), np.float32), np.ones(33, bool), 2.0, MEDIAPIPE33)
        self.assertEqual(dm.metric_joint_coverage(mp), (13, 13))


class MultiPersonTest(unittest.TestCase):
    def test_per_id_engines_floor_and_timeout(self):
        pts = np.zeros((26, 3))
        pts[[15, 16, 24, 25], 1] = 0.9  # 腳踝、腳跟 y = 0.9
        metrics = dm.DanceMetrics(timeout=1.0)
        for i in range(5):
            out = metrics.update([person(1, pts), person(2, pts + [1, 0, 0])], i / FPS)
        self.assertEqual(set(out), {1, 2})
        self.assertAlmostEqual(metrics.floor_y, 0.9 + dm.DanceMetrics.ANKLE_HEIGHT, places=3)  # 腳踝 + 腳踝高度
        self.assertIsNotNone(out[1]["height"])
        out = metrics.update([person(2, pts)], 2.0)  # ID 1 消失超過 timeout
        self.assertEqual(set(metrics.engines), {2})

    def test_estimated_feet_do_not_set_floor(self):
        pts = np.zeros((26, 3))
        pts[[15, 16, 24, 25], 1] = 0.9
        metrics = dm.DanceMetrics()
        out = metrics.update([person(1, pts, measured=False)], 0.0)
        metrics.update([person(1, pts, measured=False)], 1 / FPS)
        self.assertIsNone(metrics.floor_y)
        self.assertIn(1, out)

    def test_pose_visibility_marks_undetected_joints(self):
        """骨架模型信心值低的關節（例如沒入鏡的腳）不參與指標。"""
        pts = np.zeros((26, 3))
        pts[:, 1] = np.linspace(-0.8, 0.9, 26)
        pts[:, 0] = np.linspace(-0.3, 0.3, 26)
        pts[:, 2] = 2 + np.linspace(0, 0.2, 26)
        vis = np.ones(26, np.float32)
        vis[[13, 14, 15, 16]] = 0.1  # 膝、踝
        metrics = dm.DanceMetrics()
        for i in range(3):
            out = metrics.update([person(1, pts, visibility=vis)], i / FPS)
        self.assertIsNone(out[1]["sway"])
        detected = dm.to_h36m_detected(vis >= 0.5, HALPE26)
        self.assertFalse(detected[[dm.L_KNEE, dm.R_KNEE, dm.L_ANKLE, dm.R_ANKLE]].any())
        self.assertTrue(detected[[dm.PELVIS, dm.SPINE, dm.NECK, dm.L_WRIST]].all())


if __name__ == "__main__":
    unittest.main()
