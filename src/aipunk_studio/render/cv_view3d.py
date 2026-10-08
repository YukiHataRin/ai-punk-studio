"""OpenCV 自繪的 3D 骨架視圖（給 OpenCV 檢視器 apps/cv_viewer.py 用；Qt 介面用 ui/widgets/view3d_widget.py）。"""

import cv2
import numpy as np

from .colors import track_bgr
from .scene3d import MIN_VISIBILITY


class Orbit3DView:
    """滑鼠拖曳旋轉、滾輪縮放的 3D 視圖。座標系同相機：x 右、y 下、z 前。"""

    def __init__(self, size):
        self.size = size
        self.reset()
        self.floor_y = 1.0
        self._drag = None

    def reset(self):
        self.yaw, self.pitch, self.dist = np.deg2rad(-35), np.deg2rad(20), 4.5
        self.target = np.array([0.0, 0.3, 2.5])

    def on_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (x, y, self.yaw, self.pitch)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            x0, y0, yaw0, pitch0 = self._drag
            self.yaw = yaw0 + (x - x0) * 0.01
            self.pitch = np.clip(pitch0 + (y - y0) * 0.01, -1.5, 1.5)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.dist = np.clip(self.dist * (0.9 if flags > 0 else 1.1), 1.0, 15.0)

    def _project(self, pts):
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        v = (Rx @ Ry @ (np.asarray(pts, np.float64) - self.target).T).T
        v[:, 2] += self.dist
        f = self.size * 1.1
        z = np.maximum(v[:, 2], 1e-3)
        uv = np.stack([v[:, 0] / z * f + self.size / 2, v[:, 1] / z * f + self.size / 2], 1)
        return uv.astype(int), v[:, 2] > 0.1

    def _line(self, img, p, q, color, thick=1):
        uv, ok = self._project([p, q])
        if ok.all():
            cv2.line(img, tuple(uv[0]), tuple(uv[1]), color, thick, cv2.LINE_AA)

    def render(self, people):
        skeletons = [p.skeleton for p in people if p.skeleton is not None]
        img = np.full((self.size, self.size, 3), 24, np.uint8)

        # 地板高度跟著腳踝慢慢調整
        ankles = [sk.points[j, 1] for sk in skeletons for j in sk.fmt.ankles if sk.measured[j]]
        if ankles:
            self.floor_y = 0.9 * self.floor_y + 0.1 * (max(ankles) + 0.08)
        for k in np.arange(-2.5, 2.51, 0.5):
            self._line(img, (k, self.floor_y, 0.5), (k, self.floor_y, 6.0), (60, 60, 60))
        for k in np.arange(0.5, 6.01, 0.5):
            self._line(img, (-2.5, self.floor_y, k), (2.5, self.floor_y, k), (60, 60, 60))

        # 相機位置（原點）與視線方向
        o = np.zeros(3)
        for c in [(-0.15, -0.1, 0.3), (0.15, -0.1, 0.3), (0.15, 0.1, 0.3), (-0.15, 0.1, 0.3)]:
            self._line(img, o, c, (180, 180, 180))
        self._line(img, (-0.15, -0.1, 0.3), (0.15, -0.1, 0.3), (180, 180, 180))
        self._line(img, (-0.15, 0.1, 0.3), (0.15, 0.1, 0.3), (180, 180, 180))

        for person in people:
            color = track_bgr(person.track_id)
            dim = tuple(int(c * 0.45) for c in color)
            sk = person.skeleton
            if sk is None:
                # 只有遮罩深度的人：從地板拉一條線到 3D 位置
                if person.centroid is not None:
                    c = person.centroid
                    self._line(img, (c[0], self.floor_y, c[2]), c, color, 2)
                    uv, ok = self._project([c])
                    if ok[0]:
                        cv2.circle(img, tuple(uv[0]), 6, color, 2, cv2.LINE_AA)
                continue
            detected = (person.pose.visibility >= MIN_VISIBILITY) if person.pose is not None else np.ones(len(sk.points), bool)
            for a, b in sk.fmt.connections:
                if not (detected[a] and detected[b]):  # 沒偵測到的關節不畫
                    continue
                both = sk.measured[a] and sk.measured[b]
                self._line(img, sk.points[a], sk.points[b], color if both else dim, 2)
            uv, ok = self._project(sk.points)
            for j in range(len(uv)):
                if ok[j] and detected[j]:
                    cv2.circle(img, tuple(uv[j]), 3, color if sk.measured[j] else dim, -1, cv2.LINE_AA)

        cv2.putText(img, "3D  drag: rotate  wheel: zoom  r: reset", (10, self.size - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return img
