"""pyqtgraph OpenGL 3D 視圖：地板網格、相機位置、每人骨架、關節、ID 標籤、遮罩點雲。

滑鼠左鍵拖曳旋轉、滾輪縮放、中鍵拖曳平移（pyqtgraph 內建）。
"""

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from PySide6.QtGui import QColor

from ...render.scene3d import build_scene, rgba
from ..theme import ui_font


class View3DWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(QColor("#101b23"))
        self.floor_z = -1.0
        self.reset_view()

        self.grid = gl.GLGridItem()
        self.grid.setSize(6, 7)
        self.grid.setSpacing(0.5, 0.5)
        self.grid.setColor((60, 90, 100, 120))
        self.grid.translate(0, 3.5, self.floor_z)
        self.addItem(self.grid)

        # 相機（原點）：小視錐，朝 +y
        o, c = np.zeros(3), np.array([[-0.15, 0.3, 0.1], [0.15, 0.3, 0.1], [0.15, 0.3, -0.1], [-0.15, 0.3, -0.1]])
        cam = [o, c[0], o, c[1], o, c[2], o, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]]
        self.addItem(gl.GLLinePlotItem(pos=np.array(cam, np.float32), color=(0.7, 0.75, 0.78, 1), width=1, mode="lines"))

        self.cloud = gl.GLScatterPlotItem(size=2, pxMode=True)
        self.bones = gl.GLLinePlotItem(width=3, mode="lines", antialias=True)
        self.joints = gl.GLScatterPlotItem(size=7, pxMode=True)
        for item in (self.cloud, self.bones, self.joints):
            self.addItem(item)
        self._labels = {}
        self._font = ui_font(11)

    def reset_view(self):
        self.setCameraPosition(pos=Vector(0, 2.5, 0), distance=5.5, elevation=22, azimuth=-110)

    def show_people(self, people, point_cloud=False):
        scene = build_scene(people, point_cloud)

        if scene.ankle_heights:  # 地板跟著實測腳踝慢慢調整
            target = min(scene.ankle_heights) - 0.08
            self.floor_z = 0.9 * self.floor_z + 0.1 * target
            self.grid.resetTransform()
            self.grid.translate(0, 3.5, self.floor_z)

        empty3, empty4 = np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
        self.bones.setData(pos=scene.lines if len(scene.lines) else empty3,
                           color=scene.line_colors if len(scene.lines) else empty4)
        self.joints.setData(pos=scene.joints if len(scene.joints) else empty3,
                            color=scene.joint_colors if len(scene.joints) else empty4)
        self.cloud.setData(pos=scene.cloud if len(scene.cloud) else empty3,
                           color=scene.cloud_colors if len(scene.cloud) else empty4)

        seen = set()
        for tid, pos, text in scene.labels:
            seen.add(tid)
            item = self._labels.get(tid)
            r, g, b, _ = (rgba(tid) * 255).astype(int)
            if item is None:
                item = gl.GLTextItem(font=self._font)
                self.addItem(item)
                self._labels[tid] = item
            item.setData(pos=pos, text=text, color=QColor(r, g, b))
        for tid in list(self._labels):
            if tid not in seen:
                self.removeItem(self._labels.pop(tid))
