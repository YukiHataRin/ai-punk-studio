"""單幀流程，骨架有兩種後端（設定檔 [pose] backend）：

rtmpose（預設，由上而下）：
    RGB ─┬─► YOLO 分割 + BoT-SORT（執行緒 A）──────────────► 這一幀的遮罩、框、ID ─┐
         └─► RTMPose 逐人骨架，用「上一幀」的人框（執行緒 B）──────────────────────┤ 依 track ID 對應
    深度 ──► D2C 對齊（主執行緒）──────────────────────────────────────────────────┤
    用上一幀的框才能讓 RTMPose 與 YOLO 平行（30 fps 下人只移動幾個像素，框已放大 1.25 倍）；
    代價是新出現的人晚一幀才有骨架。設定 [pose] pipelined = false 則改為等 YOLO 完成再估計。

mediapipe（整張畫面多人）：
    RGB ─┬─► YOLO 分割 + BoT-SORT（執行緒 A）─┐
         └─► MediaPipe 骨架（執行緒 B）──────┤  兩者平行，事後以「關節落在遮罩內的比例」配對
    深度 ──► D2C 對齊（主執行緒）─────────────┤
                                             ▼
         遮罩內取深度提升 3D → 遮罩重心 3D 位置 → One Euro 平滑（以 track ID 為 key）→ FrameOutput

ID 來源：
- 分割 + 追蹤開啟：用 BoT-SORT 的 track ID。每個有 ID 的人都會輸出（沒配到骨架的只有遮罩與位置）；
  沒配到任何遮罩的骨架視為誤偵測，丟棄。
- 分割關閉或追蹤關閉：用 PoseTracker 依骨架位置配對產生 ID。
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..core.fusion import associate, depth_mask, mask_centroid, mask_points, match_previous_boxes
from ..core.dance_metrics import DanceMetrics
from ..core.geometry import Intrinsics
from ..core.lift import lift_person
from ..core.registration import DepthToRgb
from ..core.tracker import PoseTracker, SkeletonSmoother
from ..core.types import SegmentationResult, TrackedPerson


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
        self.dance = DanceMetrics()  # 每位舞者的九項動作指標

        # RTMPose 需要 YOLO 的人框；關閉分割時退回 MediaPipe
        backend = cfg["pose"].get("backend", "rtmpose")
        self.top_down = backend == "rtmpose" and segmentation
        self.pipelined = self.top_down and cfg["pose"].get("pipelined", True)
        self._prev_boxes = []  # [(track_id, box)]：上一幀的人框，給 pipelined RTMPose 用

        # 模型在各自專屬的執行緒建立與呼叫
        self._pose_thread = ThreadPoolExecutor(1, thread_name_prefix="pose")
        self.pose = self._pose_thread.submit(self._make_pose, cfg["pose"]).result()
        self._seg_thread = None
        self.segmenter = None
        if segmentation:
            from ..perception.segmentation import PersonSegmenter
            self._seg_thread = ThreadPoolExecutor(1, thread_name_prefix="segmentation")
            self.segmenter = self._seg_thread.submit(PersonSegmenter, cfg["segmentation"]).result()
            self._seg_thread.submit(self.segmenter.warmup).result()

    def _make_pose(self, params):
        if self.top_down:
            from ..perception.pose_rtm import RTMPoseEstimator
            return RTMPoseEstimator(params)
        from ..perception.pose import PoseEstimator
        return PoseEstimator(params)

    @property
    def pose_label(self):
        name = "RTMPose-m" if self.top_down else "MediaPipe"
        return f"{name}：{self.pose.device_label}"

    @property
    def tracking(self):
        return self.segmenter is not None and self.cfg["segmentation"]["tracking"]

    def process(self, frame, depth_mm, stamp) -> FrameOutput:
        t0 = time.perf_counter()
        seg_job = self._seg_thread.submit(self.segmenter, frame) if self.segmenter else None
        if self.pipelined:
            prev = self._prev_boxes
            pose_job = self._pose_thread.submit(self.pose, frame, [b for _, b in prev])
        elif not self.top_down:
            pose_job = self._pose_thread.submit(self.pose, frame)
        reg_depth = self.registration(depth_mm) if depth_mm is not None else self.registration.empty()
        seg = seg_job.result() if seg_job else SegmentationResult()
        if self.pipelined:
            poses = pose_job.result()
            matched = match_previous_boxes(prev, seg.people)
            pairs = {k: j for k, j in matched.items() if poses[k] is not None}
            self._prev_boxes = [(s.track_id, s.box) for s in seg.people]
        elif self.top_down:
            # 每個人框各估一副骨架，索引與 seg.people 一一對應（超過人數上限的為 None）
            poses = self._pose_thread.submit(self.pose, frame, [p.box for p in seg.people]).result()
            pairs = {i: i for i, p in enumerate(poses) if p is not None}
        else:
            poses = pose_job.result()
            pairs = associate(poses, seg.people, self.cfg["fusion"]["min_score"])

        t1 = time.perf_counter()
        people = [self.smoother.smooth(p, stamp) for p in self._fuse(poses, pairs, seg, reg_depth, stamp)]
        self.smoother.prune(stamp)
        metrics = self.dance.update(people, stamp)  # 用平滑後的骨架算：導數（尤其急動度）對抖動很敏感
        for p in people:
            p.metrics = metrics.get(p.track_id)
        t2 = time.perf_counter()

        return FrameOutput(frame, reg_depth, seg, sorted(people, key=lambda p: p.track_id), {
            "segmentation": seg.inference_ms,
            "pose": self.pose.inference_ms,
            "fusion": (t2 - t1) * 1000,
            "total": (t2 - t0) * 1000,
        })

    def _fuse(self, poses, pairs, seg, reg_depth, stamp):
        """pairs：{pose 索引: segment 索引}。poses 可能含 None（RTMPose 超過人數上限時）。"""
        f, sk_params = self.cfg["fusion"], self.cfg["skeleton"]
        scale = self.registration.scale
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

        valid = [i for i, p in enumerate(poses) if p is not None and (not self.top_down or i in pairs)]
        ids = self.pose_tracker.assign([poses[i] for i in valid], stamp)
        return [build(tid, poses[i], pairs.get(i)) for i, tid in zip(valid, ids)]

    def reset_tracking(self):
        """切換追蹤開關或換來源時呼叫：清掉所有 ID 與平滑狀態。"""
        self.pose_tracker.reset()
        self.smoother.reset()
        self.dance.reset()
        self._prev_boxes = []
        if self.segmenter:
            self._seg_thread.submit(self.segmenter.reset).result()

    def close(self):
        self._pose_thread.submit(self.pose.close).result()
        self._pose_thread.shutdown()
        if self._seg_thread:
            self._seg_thread.shutdown()
