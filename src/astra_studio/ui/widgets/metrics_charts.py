"""指標頁：左側是每位舞者的照片（外框為該人的代表色），右側是九項舞蹈指標的即時曲線（3 × 3），
每位舞者一條線，顏色與照片外框、畫面中的 ID 一致。

資料在背景持續累積（切到其他頁時也一樣）。為了不拖慢影像：曲線約 5 Hz 重繪、關閉反鋸齒並依視窗降採樣，
照片約 2 Hz 更新。
"""

import time
from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ...core.dance_metrics import METRICS
from ...render.colors import track_hex
from ..theme import BORDER, FAINT, MUTED, PANEL

WINDOW_S = 20.0          # 顯示最近幾秒
REDRAW_INTERVAL = 0.2    # 曲線重繪間隔（秒）
THUMB_INTERVAL = 0.5     # 照片更新間隔（秒）
THUMB_W, THUMB_H = 150, 190


def crop_person(frame, person, pad=0.12):
    """依人框（沒有時用 2D 關節範圍）從畫面裁出這個人，回傳 BGR 影像或 None。"""
    if person.segment is not None:
        x1, y1, x2, y2 = person.segment.box
    elif person.pose is not None:
        (x1, y1), (x2, y2) = person.pose.pixels.min(0), person.pose.pixels.max(0)
    else:
        return None
    h, w = frame.shape[:2]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def to_pixmap(bgr, width, height):
    """等比例縮放並置中補邊到固定大小。"""
    scale = min(width / bgr.shape[1], height / bgr.shape[0])
    small = cv2.resize(bgr, (max(1, int(bgr.shape[1] * scale)), max(1, int(bgr.shape[0] * scale))))
    canvas = np.full((height, width, 3), 16, np.uint8)
    y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy())


class PersonTile(QFrame):
    """一位舞者：照片（代表色外框）、ID、距離；離開畫面後變暗並標示「已離開」。"""

    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id
        self.color = track_hex(track_id)
        self.setObjectName("personTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        self.photo = QLabel()
        self.photo.setFixedSize(THUMB_W, THUMB_H)
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.photo, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.title)
        self.note = QLabel("")
        self.note.setObjectName("cardNote")
        layout.addWidget(self.note)
        self.set_present(True)

    def set_present(self, present, distance_text=""):
        border = self.color if present else BORDER
        self.photo.setStyleSheet(f"border: 3px solid {border}; border-radius: 6px; background: #101b23;")
        name_color = self.color if present else FAINT
        self.title.setText(f"<span style='color:{name_color}; font-weight:600'>● ID {self.track_id}</span>")
        self.note.setText(distance_text if present else "已離開")
        self.setStyleSheet(f"QFrame#personTile {{ background: {PANEL}; border-radius: 10px; }}")

    def set_photo(self, bgr):
        self.photo.setPixmap(to_pixmap(bgr, THUMB_W - 6, THUMB_H - 6))


class MetricsCharts(QWidget):
    def __init__(self):
        super().__init__()
        pg.setConfigOptions(antialias=False)  # 反鋸齒很吃 CPU，會拖慢擷取
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(12)

        # ---- 左側：人物照片 ----
        legend = QWidget()
        self.legend_layout = QVBoxLayout(legend)
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_layout.setSpacing(10)
        self.empty = QLabel("畫面中還沒有人")
        self.empty.setObjectName("caption")
        self.legend_layout.addWidget(self.empty)
        self.legend_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(legend)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(THUMB_W + 40)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        # ---- 右側：九張曲線 ----
        right = QVBoxLayout()
        hint = QLabel(f"最近 {WINDOW_S:.0f} 秒；每位舞者一條線，顏色與左側照片外框相同。滑鼠停在圖上可看指標說明。")
        hint.setObjectName("caption")
        right.addWidget(hint)
        grid = QGridLayout()
        grid.setSpacing(8)
        right.addLayout(grid, 1)
        outer.addLayout(right, 1)

        self.plots = {}
        for i, m in enumerate(METRICS):
            w = pg.PlotWidget(background="#101b23")
            w.setToolTip(f"{m.name}（{m.name_en}）：{m.description}")
            w.setTitle(f"<span style='color:{MUTED}'>{m.name}（{m.unit}）</span>", size="10pt")
            w.showGrid(x=True, y=True, alpha=0.15)
            w.setXRange(-WINDOW_S, 0, padding=0)
            w.setMouseEnabled(x=False, y=False)
            w.hideButtons()
            w.setClipToView(True)
            w.setDownsampling(auto=True, mode="peak")  # 保留尖峰的降採樣
            for axis in ("left", "bottom"):
                w.getAxis(axis).setPen(pg.mkPen(BORDER))
                w.getAxis(axis).setTextPen(pg.mkPen(FAINT))
            grid.addWidget(w, i // 3, i % 3)
            self.plots[m.key] = w

        self.history = {}  # track_id → {"t": deque, key: deque}
        self.curves = {}   # (track_id, key) → PlotDataItem
        self.tiles = {}    # track_id → PersonTile
        self._last_draw = 0.0
        self._last_thumb = 0.0

    # ---- 資料 ----
    def add_frame(self, people, t, frame=None):
        present = set()
        for p in people:
            if p.metrics is None:
                continue
            present.add(p.track_id)
            h = self.history.get(p.track_id)
            if h is None:
                h = self.history[p.track_id] = {"t": deque(), **{m.key: deque() for m in METRICS}}
            h["t"].append(t)
            for m in METRICS:
                v = p.metrics.get(m.key)
                h[m.key].append(np.nan if v is None else v)
        for tid in list(self.history):  # 丟掉超出視窗的舊資料；整條線都過期就移除
            h = self.history[tid]
            while h["t"] and t - h["t"][0] > WINDOW_S:
                for q in h.values():
                    q.popleft()
            if not h["t"]:
                self._remove(tid)
        self._update_tiles(people, present, frame)
        if self.isVisible() and time.monotonic() - self._last_draw >= REDRAW_INTERVAL:
            self.redraw(t)

    def _remove(self, tid):
        del self.history[tid]
        for m in METRICS:
            curve = self.curves.pop((tid, m.key), None)
            if curve is not None:
                self.plots[m.key].removeItem(curve)
        tile = self.tiles.pop(tid, None)
        if tile is not None:
            tile.deleteLater()

    def _update_tiles(self, people, present, frame):
        refresh_photo = frame is not None and time.monotonic() - self._last_thumb >= THUMB_INTERVAL
        if refresh_photo:
            self._last_thumb = time.monotonic()
        by_id = {p.track_id: p for p in people}
        order = sorted(self.history, key=lambda tid: (tid not in present, tid))  # 畫面中的人在前、已離開的在後
        for i, tid in enumerate(order):
            tile = self.tiles.get(tid)
            if tile is None:
                tile = self.tiles[tid] = PersonTile(tid)
            self.legend_layout.insertWidget(i, tile)
            p = by_id.get(tid)
            tile.set_present(tid in present, p.distance_text() if p is not None else "")
            if p is not None and (refresh_photo or tile.photo.pixmap().isNull()) and frame is not None:
                crop = crop_person(frame, p)
                if crop is not None:
                    tile.set_photo(crop)
        self.empty.setVisible(not self.history)

    def redraw(self, now):
        self._last_draw = time.monotonic()
        for tid, h in self.history.items():
            ts = np.array(h["t"]) - now
            for m in METRICS:
                curve = self.curves.get((tid, m.key))
                if curve is None:
                    curve = self.plots[m.key].plot(pen=pg.mkPen(track_hex(tid), width=2), connect="finite")
                    self.curves[(tid, m.key)] = curve
                curve.setData(ts, np.array(h[m.key], dtype=float))

    def clear(self):
        for tid in list(self.history):
            self._remove(tid)
