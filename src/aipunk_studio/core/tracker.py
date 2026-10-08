"""ID 指派與時間平滑。

- SkeletonSmoother：以 track ID 為 key，對每個人的 2D / 3D 關節與位置做 One Euro 平滑。
  ID 可來自 BoT-SORT（有分割時）或 PoseTracker（只跑骨架時）。
- PoseTracker：沒有 YOLO 追蹤時的備援，用髖部中心 2D 距離貪婪配對產生 ID。
"""

from dataclasses import dataclass, replace

import numpy as np

from .filters import OneEuroFilter


@dataclass
class _Filters:
    f2d: OneEuroFilter
    f3d: OneEuroFilter
    fpos: OneEuroFilter
    last_seen: float = 0.0


@dataclass
class _Track:
    id: int
    center: np.ndarray
    last_seen: float


class SkeletonSmoother:
    def __init__(self, params):
        """params：設定檔的 [smoothing] 區段。"""
        self.s = params
        self._filters: dict[int, _Filters] = {}

    def _get(self, track_id, t):
        f = self._filters.get(track_id)
        if f is None:
            s = self.s
            f = _Filters(OneEuroFilter(s["min_cutoff_2d"], s["beta_2d"]),
                         OneEuroFilter(s["min_cutoff_3d"], s["beta_3d"]),
                         OneEuroFilter(s["min_cutoff_3d"], s["beta_3d"]))
            self._filters[track_id] = f
        f.last_seen = t
        return f

    def smooth(self, person, t):
        """回傳平滑後的 TrackedPerson（不修改輸入）。"""
        f = self._get(person.track_id, t)
        pose, sk, centroid = person.pose, person.skeleton, person.centroid
        if pose is not None:
            pose = replace(pose, pixels=f.f2d(pose.pixels, t).astype(np.float32))
        if sk is not None:
            pts = f.f3d(sk.points, t).astype(np.float32)
            sk = replace(sk, points=pts)
            sk = replace(sk, distance=float(np.linalg.norm(sk.hip_center)))
        if centroid is not None:
            centroid = f.fpos(centroid, t).astype(np.float32)
        return replace(person, pose=pose, skeleton=sk, centroid=centroid)

    def set_params(self, **changes):
        """即時調整平滑參數（例如 min_cutoff_3d），套用到既有與之後的濾波器。"""
        self.s = {**self.s, **changes}
        for f in self._filters.values():
            f.f2d.min_cutoff, f.f2d.beta = self.s["min_cutoff_2d"], self.s["beta_2d"]
            for flt in (f.f3d, f.fpos):
                flt.min_cutoff, flt.beta = self.s["min_cutoff_3d"], self.s["beta_3d"]

    def prune(self, t):
        timeout = self.s["track_timeout"]
        self._filters = {k: f for k, f in self._filters.items() if t - f.last_seen < timeout}

    def reset(self):
        self._filters.clear()


class PoseTracker:
    def __init__(self, params, image_width):
        """params：設定檔的 [smoothing] 區段。"""
        self.s = params
        self.max_jump = params["match_distance"] * image_width  # 配對允許的最大位移（像素）
        self.tracks: list[_Track] = []
        self._next_id = 1

    def assign(self, poses, t):
        """回傳與 poses 等長的 ID 列表。"""
        # 先刪掉消失太久的追蹤，避免離開很久的人回來時搶回舊 ID
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen < self.s["track_timeout"]]
        centers = [p.pixels[list(p.fmt.hips)].mean(0) for p in poses]

        # 依距離由近到遠貪婪配對
        pairs = sorted((np.linalg.norm(c - tr.center), i, k)
                       for i, c in enumerate(centers) for k, tr in enumerate(self.tracks))
        assigned, used = {}, set()
        for dist, i, k in pairs:
            if dist > self.max_jump or i in assigned or k in used:
                continue
            assigned[i] = self.tracks[k]
            used.add(k)

        ids = []
        for i, c in enumerate(centers):
            tr = assigned.get(i)
            if tr is None:
                tr = _Track(self._next_id, c, t)
                self._next_id += 1
                self.tracks.append(tr)
            tr.center, tr.last_seen = c, t
            ids.append(tr.id)
        return ids

    def reset(self):
        self.tracks.clear()
