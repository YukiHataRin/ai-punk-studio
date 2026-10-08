"""九項舞蹈動作指標（每位舞者各自計算）。

移植自 Real-time Dance Aesthetics Analysis（https://github.com/YukiHataRin/realtime-dance-analysis，
MIT License，Copyright (c) 2026 Hsing-Hao Yeh, Feng-Cheng Lin, Cheng-Tai Jiang，授權全文見 licenses/）。
公式與肢體／質量權重沿用原專案 backend/dance_metrics.py 與 constants.py，調整處：

- 單位：輸入為深度相機量到的公尺座標（原專案為 MediaPipe 推估座標 ×1000 的毫米）。
  擴展度直接回報 m³（原專案為 mm³ ÷ 1e8，即 m³ × 10）。
- 時間：每幀以實際時間戳計算 Δt，幀率變動時導數仍正確（原專案固定 1/fps）。
- 重心高度：改為「重心離地高度」，地板由深度實測的腳踝／腳跟估計
  （原專案為相對髖部原點的高度；本專案座標原點在相機，相對髖部沒有意義）。地板未知時為 None。
- 曲率：κ = |v × a| / (|v|³ + v₀³)，v₀ = CURVATURE_REST_SPEED。靜止時速度只剩抖動，
  原公式除以 |v|³ 會把雜訊放大成極大的曲率；加上 v₀ 後靜止時趨近 0，快速動作時與原公式相同。
  上限 100 1/m（半徑 1 cm）；原專案以毫米計算，其上限實際上不會觸發。
- 擴展度：退化（共面）時用 qhull 的 QJ 選項，取代原專案每幀加隨機雜訊。
- 只用骨架模型偵測到的關節：沒入鏡或被擋住的關節是模型猜的，原專案照算，猜測值的抖動會被當成動作。
  - 強度、出力、急動度：只加總偵測到的肢體群組（腳不在畫面內時只算上半身）
  - 左右平衡／協調：只比較兩側都偵測到的肢體（例如只比較雙手）
  - 擴展度：只用偵測到的關節構成凸包；曲率：只平均偵測到的手腕、腳踝
  - 重心高度與晃動：需要兩腳踝都偵測到；重心以偵測到的關節計算
  - 沒有可用的關節時為 None
- 左右平衡／協調的分母加上靜止門檻（OMEGA_REST、CORR_REST）：靜止時兩側角速度只剩雜訊，原公式等於拿兩個雜訊相除，
  數值會在 0–1（協調為 −1–1）之間亂跳。加上門檻後靜止時平衡趨近 1、協調趨近 0，
  動作明顯大於門檻時與原公式相同。

指標鍵值與原專案相同，方便對照與沿用前端。
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .fusion import MIN_VISIBILITY

# ---- H36M 17 關節（與原專案 constants.py 相同的編號）----
PELVIS, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE = range(7)
SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST = range(11, 17)

LIMB_GROUPS = {
    "Trunk": [(PELVIS, SPINE), (SPINE, THORAX), (THORAX, NECK), (NECK, HEAD)],
    "L_Arm": [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST)],
    "R_Arm": [(R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)],
    "L_Leg": [(L_HIP, L_KNEE), (L_KNEE, L_ANKLE)],
    "R_Leg": [(R_HIP, R_KNEE), (R_KNEE, R_ANKLE)],
}
LIMB_WEIGHTS = {"Trunk": 0.5, "L_Leg": 0.15, "R_Leg": 0.15, "L_Arm": 0.1, "R_Arm": 0.1}

_mass = np.zeros(17)
_mass[PELVIS], _mass[SPINE], _mass[THORAX], _mass[HEAD] = 0.1, 0.2, 0.05, 0.05
_mass[[L_HIP, R_HIP]] = 0.1
_mass[[L_KNEE, R_KNEE]] = 0.05
_mass[[L_ANKLE, R_ANKLE]] = 0.02
_mass[[L_SHOULDER, R_SHOULDER]] = 0.05
_mass[[L_ELBOW, R_ELBOW]] = 0.03
_mass[[L_WRIST, R_WRIST]] = 0.01
MASS_WEIGHTS = _mass / _mass.sum()


@dataclass(frozen=True)
class MetricInfo:
    key: str
    name: str        # 中文名稱
    name_en: str     # 原專案英文名稱
    unit: str
    description: str


METRICS = (
    MetricInfo("energy", "動作強度", "Intensity", "rad²/s²", "肢體角速度平方的加權和，越大動作越激烈"),
    MetricInfo("sync_velocity", "左右平衡", "Sync – Balance", "0–1", "左右兩側角速度大小的相似度，1 為完全平衡（靜止時接近 1）"),
    MetricInfo("sync_correlation", "左右協調", "Sync – Correlation", "−1–1", "左右動作歷史的皮爾森相關，1 為同步、−1 為交替（靜止時接近 0）"),
    MetricInfo("expansion", "身體擴展", "Volume", "m³", "偵測到的關節構成的凸包體積，越大身體越舒展"),
    MetricInfo("curvature", "軌跡曲率", "Roundness", "1/m", "手腕、腳踝軌跡的平均曲率，越大動作越圓轉"),
    MetricInfo("height", "重心高度", "Stability – Height", "m", "重心離地高度（需要深度量到腳）"),
    MetricInfo("sway", "重心晃動", "Stability – Sway", "m", "重心偏離兩腳中點的水平距離（需要看到雙腳）"),
    MetricInfo("torque", "出力程度", "Effort", "rad/s²", "肢體角加速度的加權和（出力的代理指標，非實際力矩）"),
    MetricInfo("jerk", "急動度", "Smoothness", "rad²/s⁶", "角急動度平方的加權和，越低越流暢"),
)
METRIC_KEYS = tuple(m.key for m in METRICS)

# 指標實際用到的 13 個關節（其餘 4 個 H36M 點由它們推算）；臉、手指、腳掌都不參與
METRIC_JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                 "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")

CURVATURE_REST_SPEED = 0.3      # m/s：曲率的靜止門檻（靜止時手腕抖動速度約 0.05–0.2 m/s，跳舞時 1–3 m/s）
MIN_VOLUME = 1e-6               # m³（1 cm³）：小於此值視為扁平（例如沒有深度時所有關節在同一平面），回報 0
MAX_CURVATURE = 100.0           # 1/m
MIN_DT, MAX_DT = 1e-3, 0.5
OMEGA_REST = 0.5                # rad/s（每組肢體）：左右平衡的靜止門檻（靜止時骨架抖動約 0.05–0.3 rad/s）
CORR_REST = 0.05                # (rad/s)²（每組肢體）：左右協調的變異數門檻，避免雜訊之間的相關被當成協調
SIDE_PAIRS = (("L_Arm", "R_Arm"), ("L_Leg", "R_Leg"))


def metric_joint_indices(fmt):
    return [fmt.names.index(n) for n in METRIC_JOINTS]


def metric_joint_coverage(skeleton):
    """(實測的指標關節數, 13)：指標的可信度只看這 13 個關節。"""
    idx = metric_joint_indices(skeleton.fmt)
    return int(skeleton.measured[idx].sum()), len(idx)


def to_h36m(points, fmt):
    """任一骨架格式的 3D 關節 → H36M 17 關節（依關節名稱對應；頸、頭、骨盆的定義與原專案相同）。"""
    idx = {n: i for i, n in enumerate(fmt.names)}
    p = lambda name: points[idx[name]]  # noqa: E731
    out = np.zeros((17, 3), np.float64)
    l_hip, r_hip = p("left_hip"), p("right_hip")
    l_sh, r_sh = p("left_shoulder"), p("right_shoulder")
    pelvis = (l_hip + r_hip) / 2
    neck = (l_sh + r_sh) / 2
    spine = (pelvis + neck) / 2
    out[PELVIS], out[R_HIP], out[L_HIP] = pelvis, r_hip, l_hip
    out[R_KNEE], out[R_ANKLE] = p("right_knee"), p("right_ankle")
    out[L_KNEE], out[L_ANKLE] = p("left_knee"), p("left_ankle")
    out[SPINE], out[THORAX], out[NECK], out[HEAD] = spine, (spine + neck) / 2, neck, p("nose")
    out[L_SHOULDER], out[L_ELBOW], out[L_WRIST] = l_sh, p("left_elbow"), p("left_wrist")
    out[R_SHOULDER], out[R_ELBOW], out[R_WRIST] = r_sh, p("right_elbow"), p("right_wrist")
    return out


def to_h36m_detected(detected, fmt):
    """每個關節是否偵測到（bool 陣列）→ H36M 17 關節；推算的點（骨盆、脊椎、胸、頸）需要來源關節都偵測到。"""
    idx = {n: i for i, n in enumerate(fmt.names)}
    d = lambda name: bool(detected[idx[name]])  # noqa: E731
    out = np.zeros(17, bool)
    hips = d("left_hip") and d("right_hip")
    shoulders = d("left_shoulder") and d("right_shoulder")
    out[PELVIS] = hips
    out[NECK] = shoulders
    out[SPINE] = out[THORAX] = hips and shoulders
    out[HEAD] = d("nose")
    out[L_HIP], out[L_KNEE], out[L_ANKLE] = d("left_hip"), d("left_knee"), d("left_ankle")
    out[R_HIP], out[R_KNEE], out[R_ANKLE] = d("right_hip"), d("right_knee"), d("right_ankle")
    out[L_SHOULDER], out[L_ELBOW], out[L_WRIST] = d("left_shoulder"), d("left_elbow"), d("left_wrist")
    out[R_SHOULDER], out[R_ELBOW], out[R_WRIST] = d("right_shoulder"), d("right_elbow"), d("right_wrist")
    return out


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class DanceMetricsEngine:
    """單一舞者的指標計算（保留最近 history_size 幀）。

    每幀可附上 detected（骨架模型是否偵測到各關節）：沒偵測到的關節是模型猜的，
    它的抖動會被當成動作，所以各指標只用偵測到的關節與肢體；不足以計算時回傳 None。
    """

    def __init__(self, history_size=30):
        self.history_size = history_size
        self.positions = []   # [(17, 3)]
        self.detected = []    # [(17,) bool]
        self.stamps = []
        self.omega_l, self.omega_r = [], []
        self.omega_hist = []  # 最近 3 幀各肢體群組的角速度（沒偵測到的群組為 None）
        self.dt_hist = []

    @staticmethod
    def empty():
        return {k: None for k in METRIC_KEYS}

    def update(self, positions, t, floor_y=None, detected=None):
        """positions：(17, 3) H36M 公尺座標（相機座標系 y 向下）；detected：(17,) 骨架模型是否偵測到該關節，
        None 表示全部偵測到。回傳九項指標（資料不足的為 None）。"""
        self.positions.append(np.asarray(positions, np.float64))
        self.detected.append(np.ones(17, bool) if detected is None else np.asarray(detected, bool))
        self.stamps.append(t)
        if len(self.positions) > self.history_size:
            self.positions.pop(0)
            self.detected.pop(0)
            self.stamps.pop(0)
        if len(self.positions) < 2:
            return self.empty()
        dt = float(np.clip(self.stamps[-1] - self.stamps[-2], MIN_DT, MAX_DT))

        energy, omegas = self._energy(dt)
        sync_v, sync_c = self._synchronization(omegas)
        height, sway = self._stability(self.positions[-1], self.detected[-1], floor_y)
        torque, jerk = self._transition(omegas, dt)
        return {
            "energy": energy, "sync_velocity": sync_v, "sync_correlation": sync_c,
            "expansion": self._expansion(self.positions[-1], self.detected[-1]), "curvature": self._curvature(),
            "height": height, "sway": sway, "torque": torque, "jerk": jerk,
        }

    def _energy(self, dt):
        """1. 動作強度：肢體向量相鄰兩幀夾角 → 角速度 ω，E = Σ w·ω²（只計兩幀都偵測到的肢體群組）。"""
        pos_t, pos_prev = self.positions[-1], self.positions[-2]
        seen = self.detected[-1] & self.detected[-2]
        total, omegas, any_group = 0.0, {}, False
        for group, limbs in LIMB_GROUPS.items():
            if not all(seen[a] and seen[b] for a, b in limbs):
                omegas[group] = None
                continue
            any_group = True
            w = LIMB_WEIGHTS[group]
            omega_sum = 0.0
            for a, b in limbs:
                cos = np.clip(np.dot(_unit(pos_t[b] - pos_t[a]), _unit(pos_prev[b] - pos_prev[a])), -1.0, 1.0)
                omega = np.arccos(cos) / dt
                total += (w / len(limbs)) * omega ** 2
                omega_sum += omega
            omegas[group] = omega_sum / len(limbs)
        return (float(total) if any_group else None), omegas

    def _synchronization(self, omegas):
        """2. 左右平衡（大小相似度）與 3. 左右協調（歷史相關係數），只比較兩側都偵測到的肢體。"""
        pairs = [(l, r) for l, r in SIDE_PAIRS if omegas[l] is not None and omegas[r] is not None]
        if not pairs:
            return None, None
        wl = sum(omegas[l] for l, _ in pairs)
        wr = sum(omegas[r] for _, r in pairs)
        self.omega_l.append(wl)
        self.omega_r.append(wr)
        if len(self.omega_l) > self.history_size:
            self.omega_l.pop(0)
            self.omega_r.pop(0)
        n = len(pairs)  # 兩側角速度是 n 組肢體的加總：雜訊的大小與變異數都隨 n 增加，門檻跟著放大
        balance = 1.0 - abs(wl - wr) / (max(wl, wr) + OMEGA_REST * n)
        corr = 0.0
        if len(self.omega_l) >= 2:
            l, r = np.array(self.omega_l), np.array(self.omega_r)
            cov = np.mean((l - l.mean()) * (r - r.mean()))
            rest = CORR_REST * n
            corr = float(cov / np.sqrt((l.var() + rest) * (r.var() + rest)))
        return float(balance), corr

    @staticmethod
    def _expansion(positions, detected):
        """4. 身體擴展：偵測到的關節構成的凸包體積（m³）；少於 4 個關節時為 None。"""
        pts = positions[detected]
        if len(pts) < 4:
            return None
        try:
            volume = float(ConvexHull(pts, qhull_options="QJ").volume)
        except (QhullError, ValueError):
            return 0.0
        return volume if volume >= MIN_VOLUME else 0.0

    def _curvature(self):
        """5. 軌跡曲率：手腕、腳踝的 κ = |v × a| / (|v|³ + v₀³) 平均（只計連續三幀都偵測到的關節）。"""
        if len(self.positions) < 3:
            return None
        t0, t1, t2 = self.stamps[-3:]
        dt1, dt2 = max(t1 - t0, MIN_DT), max(t2 - t1, MIN_DT)
        seen = self.detected[-1] & self.detected[-2] & self.detected[-3]
        joints = [j for j in (L_WRIST, R_WRIST, L_ANKLE, R_ANKLE) if seen[j]]
        if not joints:
            return None
        total = 0.0
        for j in joints:
            p0, p1, p2 = (self.positions[k][j] for k in (-3, -2, -1))
            v1, v2 = (p1 - p0) / dt1, (p2 - p1) / dt2
            a = (v2 - v1) / ((dt1 + dt2) / 2)
            v = (v1 + v2) / 2
            speed = np.linalg.norm(v)
            total += min(np.linalg.norm(np.cross(v, a)) / (speed ** 3 + CURVATURE_REST_SPEED ** 3), MAX_CURVATURE)
        return float(total / len(joints))

    @staticmethod
    def _stability(positions, detected, floor_y):
        """6. 重心離地高度與 7. 重心晃動（重心與兩腳中點在水平面 xz 的距離）。

        需要兩腳踝都偵測到（看不到腳時重心與支撐點都不可靠）；重心只用偵測到的關節、質量權重重新正規化。
        """
        if not (detected[L_ANKLE] and detected[R_ANKLE]):
            return None, None
        weights = MASS_WEIGHTS * detected
        com = weights @ positions / weights.sum()
        base = (positions[L_ANKLE] + positions[R_ANKLE]) / 2
        sway = float(np.hypot(com[0] - base[0], com[2] - base[2]))
        height = None if floor_y is None else float(floor_y - com[1])  # y 向下：地板 y 較大
        return height, sway

    def _transition(self, omegas, dt):
        """8. 出力程度（角加速度）與 9. 急動度（角急動度平方），只計連續三幀都偵測到的肢體群組。"""
        self.omega_hist.append(omegas)
        self.dt_hist.append(dt)
        if len(self.omega_hist) > 3:
            self.omega_hist.pop(0)
            self.dt_hist.pop(0)
        if len(self.omega_hist) < 3:
            return None, None
        dt1, dt2 = self.dt_hist[-2], self.dt_hist[-1]
        torque = jerk_cost = 0.0
        any_group = False
        for group, w in LIMB_WEIGHTS.items():
            w0, w1, w2 = (h[group] for h in self.omega_hist)
            if w0 is None or w1 is None or w2 is None:
                continue
            any_group = True
            alpha1, alpha2 = (w1 - w0) / dt1, (w2 - w1) / dt2
            jerk = (alpha2 - alpha1) / ((dt1 + dt2) / 2)
            torque += w * abs(alpha2)
            jerk_cost += w * jerk ** 2
        if not any_group:
            return None, None
        return float(torque), float(jerk_cost)


class DanceMetrics:
    """多人版：每個追蹤 ID 一個 DanceMetricsEngine，並共同估計地板高度。"""

    FLOOR_JOINTS = ("left_ankle", "right_ankle")  # 只用指標關節
    ANKLE_HEIGHT = 0.08  # 腳踝關節中心離地約 8 cm（成人）

    def __init__(self, history_size=30, timeout=1.0):
        self.history_size = history_size
        self.timeout = timeout
        self.engines: dict[int, DanceMetricsEngine] = {}
        self.last_seen: dict[int, float] = {}
        self.floor_y = None

    def _update_floor(self, people):
        """地板 = 深度實測腳踝中最低點（y 最大）+ 腳踝高度 的移動平均；只用實測值，推估的骨架不參與。"""
        lows = []
        for p in people:
            sk = p.skeleton
            if sk is None:
                continue
            names = sk.fmt.names
            ys = [sk.points[i, 1] for i, n in enumerate(names) if n in self.FLOOR_JOINTS and sk.measured[i]]
            if ys:
                lows.append(max(ys))
        if lows:
            y = max(lows) + self.ANKLE_HEIGHT
            self.floor_y = y if self.floor_y is None else 0.95 * self.floor_y + 0.05 * y

    def update(self, people, t):
        """回傳 {track_id: 指標 dict}；沒有骨架的人不計算。"""
        self._update_floor(people)
        out = {}
        for p in people:
            if p.skeleton is None:
                continue
            engine = self.engines.get(p.track_id)
            if engine is None:
                engine = self.engines[p.track_id] = DanceMetricsEngine(self.history_size)
            self.last_seen[p.track_id] = t
            detected = None
            if p.pose is not None and p.pose.fmt is p.skeleton.fmt:
                detected = to_h36m_detected(p.pose.visibility >= MIN_VISIBILITY, p.pose.fmt)
            out[p.track_id] = engine.update(to_h36m(p.skeleton.points, p.skeleton.fmt), t, self.floor_y, detected)
        for tid in [k for k, s in self.last_seen.items() if t - s > self.timeout]:
            self.engines.pop(tid, None)
            self.last_seen.pop(tid, None)
        return out

    def reset(self):
        self.engines.clear()
        self.last_seen.clear()
        self.floor_y = None
