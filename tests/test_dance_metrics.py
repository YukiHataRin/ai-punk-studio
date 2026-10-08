"""舞蹈指標測試：用解析解已知的合成動作驗證九項指標（不需要相機與模型）。"""

import unittest

import numpy as np

from aipunk_studio.core import dance_metrics as dm
from aipunk_studio.core.skeleton_format import HALPE26, MEDIAPIPE33
from aipunk_studio.core.types import Skeleton3D, TrackedPerson

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
        self.assertLess(m["sync_velocity"], 0.01)  # 只有左邊動 → 完全不平衡

    def test_mirrored_limbs_are_synchronized(self):
        def frames(phase):
            out = []
            for i in range(40):
                speed = 1 + np.sin(i / 3)  # 速度隨時間變化，相關係數才有意義
                ang_l = 0.4 * np.sin(i / 5)
                ang_r = 0.4 * np.sin(i / 5 + phase)
                p = rotate_forearm(standing(), dm.L_ELBOW, dm.L_WRIST, ang_l * speed)
                out.append(rotate_forearm(p, dm.R_ELBOW, dm.R_WRIST, -ang_r * speed))
            return out
        same = run(dm.DanceMetricsEngine(), frames(0.0))
        self.assertGreater(same["sync_correlation"], 0.95)
        self.assertGreater(same["sync_velocity"], 0.95)

    def test_expansion_unit_cube(self):
        p = np.full((17, 3), 0.5)  # 其餘關節在內部
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        p[:8] = corners
        self.assertAlmostEqual(dm.DanceMetricsEngine._expansion(p), 1.0, places=6)
        self.assertEqual(dm.DanceMetricsEngine._expansion(np.zeros((17, 3))), 0.0)  # 退化不會出錯
        flat = np.random.default_rng(1).normal(size=(17, 3))
        flat[:, 2] = 2.0  # 沒有深度時所有關節在同一深度平面 → 扁平
        self.assertEqual(dm.DanceMetricsEngine._expansion(flat), 0.0)

    def test_height_above_floor_and_sway(self):
        p = standing()
        com = dm.MASS_WEIGHTS @ p
        height, sway = dm.DanceMetricsEngine._stability(p, floor_y=0.93)
        self.assertAlmostEqual(height, 0.93 - com[1])
        self.assertAlmostEqual(sway, 0.0, places=6)  # 左右對稱，重心正好在兩腳中點上方
        leaning = p.copy()
        leaning[[dm.L_ANKLE, dm.R_ANKLE], 0] += 0.2  # 兩腳往右移 20 cm
        _, sway = dm.DanceMetricsEngine._stability(leaning, floor_y=0.93)
        # 腳踝本身也有質量，移動腳會帶著重心移一點：0.2 × (1 − 兩腳踝質量權重)
        self.assertAlmostEqual(sway, 0.2 * (1 - 2 * dm.MASS_WEIGHTS[dm.L_ANKLE]), places=6)
        self.assertIsNone(dm.DanceMetricsEngine._stability(p, floor_y=None)[0])

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


def person(tid, points, measured=True):
    return TrackedPerson(tid, skeleton=Skeleton3D(points.astype(np.float32), np.full(26, measured), 2.0, HALPE26))


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


if __name__ == "__main__":
    unittest.main()
