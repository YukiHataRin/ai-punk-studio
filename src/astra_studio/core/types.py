"""各層之間傳遞的資料型別。"""

from dataclasses import dataclass, field

import numpy as np

from .skeleton_format import MEDIAPIPE33, SkeletonFormat


@dataclass
class PoseObservation:
    """骨架模型對一個人的 2D 輸出。K = fmt.size（MediaPipe 33、RTMPose Halpe26 26）。"""
    pixels: np.ndarray                 # (K, 2) RGB 影像像素座標
    visibility: np.ndarray             # (K,) 可見度／信心值
    world: np.ndarray | None = None    # (K, 3) MediaPipe world landmarks（公尺，髖部為原點）；RTMPose 沒有
    fmt: SkeletonFormat = MEDIAPIPE33


@dataclass
class SegmentedPerson:
    """YOLO 分割對一個人的輸出。"""
    mask: np.ndarray        # (H, W) bool，RGB 影像解析度
    box: np.ndarray         # (4,) x1, y1, x2, y2（像素）
    confidence: float
    track_id: int | None = None  # 未開追蹤時為 None


@dataclass
class SegmentationResult:
    people: list[SegmentedPerson] = field(default_factory=list)
    inference_ms: float = 0.0


@dataclass
class Skeleton3D:
    points: np.ndarray    # (K, 3) RGB 相機座標系，公尺
    measured: np.ndarray  # (K,) bool，True = 由深度實測，False = 推估
    distance: float       # 髖部中心距離（公尺）
    fmt: SkeletonFormat = MEDIAPIPE33

    @property
    def hip_center(self):
        return self.points[list(self.fmt.hips)].mean(0)


@dataclass
class TrackedPerson:
    """融合後的一個人：分割、骨架、3D 位置都可能缺（例如 YOLO 抓到人但 MediaPipe 沒抓到骨架）。"""
    track_id: int
    segment: SegmentedPerson | None = None
    pose: PoseObservation | None = None
    skeleton: Skeleton3D | None = None
    centroid: np.ndarray | None = None  # (3,) 遮罩深度中位數換算的 3D 位置（公尺）
    cloud: np.ndarray | None = None     # (N, 3) 遮罩內深度點雲（公尺，可選）

    @property
    def distance(self):
        if self.skeleton is not None:
            return self.skeleton.distance
        if self.centroid is not None:
            return float(np.linalg.norm(self.centroid))
        return None

    @property
    def distance_measured(self):
        """距離是否來自實際深度（有實測關節或遮罩深度）；False 表示整個人都是推估。"""
        return self.centroid is not None or (self.skeleton is not None and bool(self.skeleton.measured.any()))

    def distance_text(self, unit=" m"):
        """距離文字；推估值前面加 ≈。量不到任何距離時回傳空字串。"""
        d = self.distance
        if d is None:
            return ""
        return f"{'' if self.distance_measured else '≈'}{d:.2f}{unit}"

    @property
    def measured_ratio(self):
        """骨架中由深度實測的關節比例；沒有骨架時為 None。"""
        return None if self.skeleton is None else float(self.skeleton.measured.mean())
