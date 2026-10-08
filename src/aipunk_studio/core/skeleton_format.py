"""骨架關節格式定義。各模組透過這裡取得關節名稱、連線與關鍵關節索引，不寫死特定模型的編號。

- MEDIAPIPE33：MediaPipe Pose Landmarker 的 33 點（含臉部 11 點、手指、腳掌）
- HALPE26：RTMPose（body7 Halpe26）的 26 點（COCO 17 點 + 頭頂、頸、骨盆、腳趾、腳跟）
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SkeletonFormat:
    name: str
    names: tuple[str, ...]
    connections: tuple[tuple[int, int], ...]
    shoulders: tuple[int, int]   # 左、右肩
    hips: tuple[int, int]        # 左、右髖
    ankles: tuple[int, int]      # 左、右踝
    head: int                    # 標籤錨點（頭部）
    face: frozenset[int]         # 臉部點：畫小一點、3D 視圖省略

    @property
    def size(self):
        return len(self.names)

    @property
    def torso(self):
        """軀幹四點，順序可直接畫成多邊形：左肩、右肩、右髖、左髖。"""
        return (self.shoulders[0], self.shoulders[1], self.hips[1], self.hips[0])

    @property
    def body(self):
        """不含臉部的關節索引（配對、3D 顯示用）。"""
        return tuple(i for i in range(self.size) if i not in self.face)


MEDIAPIPE33 = SkeletonFormat(
    name="mediapipe33",
    names=("nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
           "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
           "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
           "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
           "left_foot_index", "right_foot_index"),
    connections=((0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
                 (11, 13), (11, 23), (12, 14), (12, 24), (13, 15), (14, 16), (15, 17), (15, 19), (15, 21),
                 (16, 18), (16, 20), (16, 22), (17, 19), (18, 20), (23, 24), (23, 25), (24, 26), (25, 27),
                 (26, 28), (27, 29), (27, 31), (28, 30), (28, 32), (29, 31), (30, 32)),
    shoulders=(11, 12), hips=(23, 24), ankles=(27, 28), head=0,
    face=frozenset(range(1, 11)),  # 鼻子（0）保留，用來配對與當標籤錨點
)

HALPE26 = SkeletonFormat(
    name="halpe26",
    names=("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
           "right_knee", "left_ankle", "right_ankle", "head", "neck", "hip", "left_big_toe",
           "right_big_toe", "left_small_toe", "right_small_toe", "left_heel", "right_heel"),
    connections=((0, 1), (0, 2), (1, 3), (2, 4), (17, 18), (0, 18),            # 頭、頸
                 (5, 6), (5, 11), (6, 12), (11, 12), (18, 19),                 # 軀幹
                 (5, 7), (7, 9), (6, 8), (8, 10),                              # 手臂
                 (11, 13), (13, 15), (12, 14), (14, 16),                       # 腿
                 (15, 24), (15, 20), (20, 22), (16, 25), (16, 21), (21, 23)),  # 腳
    shoulders=(5, 6), hips=(11, 12), ankles=(15, 16), head=17,
    face=frozenset({1, 2, 3, 4}),
)

FORMATS = {f.name: f for f in (MEDIAPIPE33, HALPE26)}
