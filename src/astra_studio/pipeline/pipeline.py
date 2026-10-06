"""單幀流程：

    RGB ─┬─► YOLO 分割 + BoT-SORT（執行緒 A）─┐
         └─► MediaPipe 骨架（執行緒 B）──────┤
    深度 ──► D2C 對齊（主執行緒，與上面平行）──┤
                                             ▼
         融合：骨架 ↔ 遮罩配對 → 遮罩內取深度提升 3D → 遮罩重心 3D 位置
                                             ▼
         One Euro 平滑（以 track ID 為 key）→ FrameOutput

ID 來源：
- 分割 + 追蹤開啟：用 BoT-SORT 的 track ID。每個有 ID 的人都會輸出（沒配到骨架的只有遮罩與位置）；
  沒配到任何遮罩的骨架視為誤偵測，丟棄。
- 分割關閉或追蹤關閉：用 PoseTracker 依骨架位置配對產生 ID。
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..core.fusion import associate, depth_mask, mask_centroid, mask_points
from ..core.geometry import Intrinsics
from ..core.lift import lift_person
from ..core.registration import DepthToRgb
from ..core.tracker import PoseTracker, SkeletonSmoother
from ..core.types import SegmentationResult, TrackedPerson
from ..perception.pose import PoseEstimator


@dataclass
class FrameOutput:
    frame: np.ndarray
    reg_depth: np.ndarray
    segmentation: SegmentationResult
    people: list[TrackedPerson] = field(default_factory=list)
    timings: dict = field(default_factory=dict)   # 各階段耗時（ms）


class Pipeline:
    def __init__(self, cfg, segmentation=True):
        self.cfg = cfg
        self.registration = DepthToRgb(cfg)
        self.rgb_k = Intrinsics.from_config(cfg["rgb"])
        self.pose_tracker = PoseTracker(cfg["smoothing"], cfg["rgb"]["width"])
        self.smoother = SkeletonSmoother(cfg["smoothing"])

        # 模型在各自專屬的執行緒建立與呼叫
        self._pose_thread = ThreadPoolExecutor(1, thread_name_prefix="pose")
        self.pose = self._pose_thread.submit(PoseEstimator, cfg["pose"]).result()
        self._seg_thread = None
        self.segmenter = None
        if segmentation:
            from ..perception.segmentation import PersonSegmenter
            self._seg_thread = ThreadPoolExecutor(1, thread_name_prefix="segmentation")
            self.segmenter = self._seg_thread.submit(PersonSegmenter, cfg["segmentation"]).result()
            self._seg_thread.submit(self.segmenter.warmup).result()

    @property
    def tracking(self):
        return self.segmenter is not None and self.cfg["segmentation"]["tracking"]

    def process(self, frame, depth_mm, stamp) -> FrameOutput:
        t0 = time.perf_counter()
        seg_job = self._seg_thread.submit(self.segmenter, frame) if self.segmenter else None
        pose_job = self._pose_thread.submit(self.pose, frame)
        reg_depth = self.registration(depth_mm) if depth_mm is not None else self.registration.empty()
        seg = seg_job.result() if seg_job else SegmentationResult()
        poses = pose_job.result()

        t1 = time.perf_counter()
        people = [self.smoother.smooth(p, stamp) for p in self._fuse(poses, seg, reg_depth, stamp)]
        self.smoother.prune(stamp)
        t2 = time.perf_counter()

        return FrameOutput(frame, reg_depth, seg, sorted(people, key=lambda p: p.track_id), {
            "segmentation": seg.inference_ms,
            "pose": self.pose.inference_ms,
            "fusion": (t2 - t1) * 1000,
            "total": (t2 - t0) * 1000,
        })

    def _fuse(self, poses, seg, reg_depth, stamp):
        f, sk_params = self.cfg["fusion"], self.cfg["skeleton"]
        scale = self.registration.scale
        pairs = associate(poses, seg.people, f["min_score"])
        masks = [depth_mask(s, reg_depth.shape, f["mask_erode_px"]) for s in seg.people]

        reg_k = self.registration.rgb_k

        def build(track_id, pose, j):
            segment = seg.people[j] if j is not None else None
            mask = masks[j] if j is not None else None
            lift_mask = mask if f["use_mask"] else None
            sk = lift_person(pose, reg_depth, scale, self.rgb_k, sk_params, lift_mask) if pose is not None else None
            centroid = cloud = None
            if mask is not None:
                centroid = mask_centroid(mask, reg_depth, reg_k)
                if f["point_cloud"]:
                    cloud = mask_points(mask, reg_depth, reg_k)
            return TrackedPerson(track_id, segment, pose, sk, centroid, cloud)

        if self.tracking:
            pose_of = {j: i for i, j in pairs.items()}
            return [build(s.track_id, poses[pose_of[j]] if j in pose_of else None, j)
                    for j, s in enumerate(seg.people) if s.track_id is not None]

        ids = self.pose_tracker.assign(poses, stamp)
        return [build(tid, pose, pairs.get(i)) for i, (pose, tid) in enumerate(zip(poses, ids))]

    def reset_tracking(self):
        """切換追蹤開關或換來源時呼叫：清掉所有 ID 與平滑狀態。"""
        self.pose_tracker.reset()
        self.smoother.reset()
        if self.segmenter:
            self._seg_thread.submit(self.segmenter.reset).result()

    def close(self):
        self._pose_thread.submit(self.pose.close).result()
        self._pose_thread.shutdown()
        if self._seg_thread:
            self._seg_thread.shutdown()
