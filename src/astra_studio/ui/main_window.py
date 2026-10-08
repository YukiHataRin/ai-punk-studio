"""主視窗：上方狀態列、左側主畫面（四種模式）+ 人物列、右側設定面板、下方狀態與截圖。"""

import copy
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from ..config import PROJECT_ROOT, resolve
from ..io.stream import StreamServer
from ..pipeline.worker import PipelineWorker
from ..sensors.discovery import apply_camera, discover_cameras, select_camera
from .camera_discovery import CameraDiscovery
from ..render.overlay import draw_contours, draw_depth, draw_masks, draw_people
from .theme import BAD, FAINT, OK, WARN
from .widgets.inspector import PIPELINE_KEYS, Inspector
from .widgets.metrics_charts import MetricsCharts
from .widgets.metrics_table import MetricsTable
from .widgets.viewport import Viewport

MODES = (("overlay", "疊圖"), ("split", "並排 3D"), ("depth", "深度"), ("skeleton", "僅骨架"), ("contour", "僅輪廓"))


class MainWindow(QMainWindow):
    def __init__(self, cfg, segmentation=True, worker_factory=PipelineWorker, enable_3d=True, playback=None,
                 camera_discover=discover_cameras, initial_camera=None):
        super().__init__()
        self.cfg = cfg
        self.segmentation = segmentation
        self.worker_factory = worker_factory
        self.worker = None
        self.playback = playback
        self.camera_discover = camera_discover
        self.initial_camera = initial_camera  # CLI --camera：名稱、裝置 ID 或編號
        self.discovery = None
        self.stream = None  # WebSocket 伺服器（跨開始／停止保留）
        self.last_has_depth = True
        self.pending_start = False  # 按下開始時攝影機清單還沒好，等列舉完再開始
        self.recording_dir = None
        self.close_requested = False
        self.last_image = None
        self.mode = "overlay"
        self.display = {"layer_mask": True, "layer_skeleton": True, "layer_labels": True,
                        "opacity": cfg["segmentation"]["opacity"], "point_cloud": cfg["fusion"]["point_cloud"],
                        "record_images": True}
        self.pipeline_settings = {}

        self.setWindowTitle("Astra Studio")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 18, 24, 14)
        layout.setSpacing(14)

        # ---- 上方狀態列 ----
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        brand = QLabel("●  Astra Studio")
        brand.setObjectName("brand")
        eyebrow = QLabel("RGB-D 人體分割  ·  多人 3D 骨架  ·  ID 追蹤")
        eyebrow.setObjectName("eyebrow")
        titles.addWidget(brand)
        titles.addWidget(eyebrow)
        header.addLayout(titles)
        header.addStretch()
        self.chips = {}
        for key in ("rgb", "depth", "fps"):
            chip = QLabel()
            chip.setObjectName("chip")
            chip.setTextFormat(Qt.TextFormat.RichText)
            header.addWidget(chip)
            self.chips[key] = chip
        self.set_chips(False, False, None)
        layout.addLayout(header)

        # ---- 工作區 ----
        workspace = QHBoxLayout()
        workspace.setSpacing(18)
        overview = QWidget()  # 「總覽」分頁
        left = QVBoxLayout(overview)
        left.setContentsMargins(0, 8, 0, 0)
        left.setSpacing(10)

        modes = QHBoxLayout()
        modes.setSpacing(0)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for key, text in MODES:
            button = QPushButton(text)
            button.setObjectName("mode")
            button.setCheckable(True)
            button.setChecked(key == self.mode)
            button.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            modes.addWidget(button)
        modes.addStretch()
        self.view_hint = QLabel("")
        self.view_hint.setObjectName("caption")
        modes.addWidget(self.view_hint)
        left.addLayout(modes)

        views = QHBoxLayout()
        views.setSpacing(10)
        self.viewport = Viewport()
        views.addWidget(self.viewport, 3)
        self.view3d = None
        if enable_3d:
            from .widgets.view3d_widget import View3DWidget
            self.view3d = View3DWidget()
            self.view3d.setMinimumWidth(360)
            views.addWidget(self.view3d, 2)
        left.addLayout(views, 1)

        people_title = QLabel("畫面中的人與動作指標")
        people_title.setObjectName("section")
        left.addWidget(people_title)
        self.metrics_table = MetricsTable()
        left.addWidget(self.metrics_table)

        self.charts = MetricsCharts()  # 「指標」分頁
        self.tabs = QTabWidget()
        self.tabs.addTab(overview, "總覽")
        self.tabs.addTab(self.charts, "指標")
        workspace.addWidget(self.tabs, 1)

        self.inspector = Inspector(cfg)
        self.inspector.changed.connect(self.on_setting)
        self.inspector.start_clicked.connect(self.toggle)
        self.inspector.refresh_clicked.connect(self.refresh_cameras)
        if playback:
            self.inspector.set_playback(playback.name)
        elif camera_discover is not None:
            QTimer.singleShot(0, self.refresh_cameras)
        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.inspector)
        workspace.addWidget(scroll)
        layout.addLayout(workspace, 1)

        # ---- 下方 ----
        footer = QHBoxLayout()
        self.status = QLabel("準備就緒")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.record_button = QPushButton("● 錄製")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_recording)
        footer.addWidget(self.record_button)
        self.snapshot_button = QPushButton("截圖")
        self.snapshot_button.clicked.connect(self.save_snapshot)
        footer.addWidget(self.snapshot_button)
        layout.addLayout(footer)

        menu = self.menuBar().addMenu("檔案")
        save = QAction("截圖…", self)
        save.setShortcut("Ctrl+S")
        save.triggered.connect(self.save_snapshot)
        menu.addAction(save)
        close = QAction("關閉", self)
        close.setShortcut("Ctrl+Q")
        close.triggered.connect(self.close)
        menu.addAction(close)

        self.set_mode(self.mode)
        if cfg["stream"]["enabled"]:
            self.set_streaming(True)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    # ---- 控制 ----
    def set_mode(self, mode):
        self.mode = mode
        if self.view3d is not None:
            self.view3d.setVisible(mode == "split")
        hints = {"split": "3D：左鍵拖曳旋轉、滾輪縮放", "depth": "檢查深度與 RGB 是否對齊，可在右側微調"}
        self.view_hint.setText(hints.get(mode, ""))

    def select_mode(self, mode):
        """程式切換模式時同步按鈕狀態。"""
        self.mode_buttons[mode].setChecked(True)  # 按鈕群組為互斥，其餘自動取消
        self.set_mode(mode)

    # ---- WebSocket 串流 ----
    def set_streaming(self, enabled):
        if enabled and self.stream is None:
            s = self.cfg["stream"]
            try:
                self.stream = StreamServer(s["host"], s["port"], s).start()
            except RuntimeError as error:
                self.status.setText(str(error))
                self.inspector.stream_enabled_box.blockSignals(True)
                self.inspector.stream_enabled_box.setChecked(False)
                self.inspector.stream_enabled_box.blockSignals(False)
                return
            if self.worker:
                self.stream.start_session(self.worker.cfg, self.last_has_depth)
        elif not enabled and self.stream is not None:
            stream, self.stream = self.stream, None
            if self.worker:
                self.worker.runner.publisher = None
            stream.stop()
        if self.worker:
            self.worker.runner.publisher = self.stream
        self.update_stream_info()

    def update_stream_info(self):
        if self.stream is None:
            self.inspector.stream_info.setText("未啟用")
        else:
            lan = "" if self.stream.host in ("127.0.0.1", "localhost") else "（區域網路可連線）"
            self.inspector.stream_info.setText(f"{self.stream.url}{lan} · {self.stream.client_count} 個用戶端")

    def on_setting(self, key, value):
        if key == "stream_enabled":
            self.set_streaming(value)
            return
        if key in PIPELINE_KEYS:
            self.pipeline_settings[key] = value
            if self.worker:
                self.worker.update(**{key: value})
        if key in self.display:
            self.display[key] = value

    # ---- 攝影機 ----
    def refresh_cameras(self):
        if self.discovery or self.worker or self.camera_discover is None:
            return
        self.inspector.source.setText("尋找攝影機…")
        self.inspector.refresh.setEnabled(False)
        self.discovery = CameraDiscovery(self.camera_discover, self)
        self.discovery.ready.connect(self.on_cameras)
        self.discovery.failed.connect(self.inspector.source.setText)
        self.discovery.finished.connect(self.on_discovery_finished)
        self.discovery.start()

    def on_cameras(self, cameras):
        preferred = select_camera(cameras, self.initial_camera) if self.initial_camera is not None else None
        self.inspector.set_cameras(cameras, preferred)

    def on_discovery_finished(self):
        discovery, self.discovery = self.discovery, None
        discovery.deleteLater()
        self.inspector.refresh.setEnabled(self.worker is None and not self.playback)
        if self.close_requested:
            self.close()
        elif self.pending_start:
            self.pending_start = False
            self.start()

    def worker_config(self):
        cfg = copy.deepcopy(self.cfg)
        cam = None if self.playback else self.inspector.selected_camera()
        if cam is not None:
            apply_camera(cfg, cam)
        choice = self.inspector.pose_model()
        if choice == "rtmpose":
            cfg["pose"]["backend"] = "rtmpose"
        else:
            cfg["pose"]["backend"] = "mediapipe"
            cfg["pose"]["model"] = str(resolve(f"models/pose_landmarker_{choice}.task"))
        s = self.pipeline_settings
        cfg["segmentation"]["confidence"] = s.get("confidence", cfg["segmentation"]["confidence"])
        cfg["segmentation"]["tracking"] = s.get("tracking", cfg["segmentation"]["tracking"])
        cfg["fusion"]["use_mask"] = s.get("use_mask", cfg["fusion"]["use_mask"])
        cfg["fusion"]["point_cloud"] = s.get("point_cloud", cfg["fusion"]["point_cloud"])
        cfg["smoothing"]["min_cutoff_3d"] = s.get("min_cutoff_3d", cfg["smoothing"]["min_cutoff_3d"])
        if "extrinsic_x" in s:
            cfg["extrinsics"]["translation"][0] = s["extrinsic_x"]
        return cfg

    def toggle(self):
        if self.worker:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.worker:
            return
        if self.discovery is not None:  # 先等攝影機清單，才知道要開哪一台
            self.pending_start = True
            self.status.setText("尋找攝影機中，找到後自動開始…")
            return
        self.viewport.clear()
        self.charts.clear()
        self.status.setText("啟動中…")
        source_factory = None
        if self.playback:
            from ..sensors.playback import PlaybackSource
            source_factory = lambda cfg, path=self.playback: PlaybackSource(path)  # noqa: E731
        self.worker = self.worker_factory(self.worker_config(), self.segmentation, self,
                                          source_factory=source_factory, publisher=self.stream)
        self.worker.status.connect(self.status.setText)
        self.worker.recording.connect(self.on_recording)
        self.worker.ready.connect(self.inspector.device.setText)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.inspector.set_running(True)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("停止中…")

    def toggle_recording(self):
        if not self.worker:
            self.record_button.setChecked(False)
            self.status.setText("請先按「開始」再錄製")
            return
        if self.recording_dir:
            self.worker.update(record=None)
        else:
            self.worker.update(record={"with_images": self.display["record_images"]})
        self.record_button.setEnabled(False)  # 等 worker 回報後再恢復

    def on_recording(self, path, active):
        self.recording_dir = path if active else None
        self.record_button.setEnabled(True)
        self.record_button.setChecked(active)
        self.record_button.setText("■ 停止錄製" if active else "● 錄製")
        if not active:
            self.status.setText(f"錄製已儲存：{path}")

    def on_error(self, message):
        self.status.setText(message)
        self.status.setToolTip(message)

    def on_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.recording_dir = None
        self.record_button.setEnabled(True)
        self.record_button.setChecked(False)
        self.record_button.setText("● 錄製")
        self.inspector.set_running(False)
        self.set_chips(False, False, None)
        if not self.status.toolTip():
            self.status.setText("已停止")
        self.status.setToolTip("")
        if self.close_requested:
            self.close()

    # ---- 畫面更新 ----
    def set_chips(self, rgb, depth, fps, has_depth=True):
        def dot(ok):
            return f"<span style='color:{OK if ok else (BAD if self.worker else FAINT)}'>●</span>"
        self.chips["rgb"].setText(f"{dot(rgb)} RGB 1280×720")
        if has_depth:
            self.chips["depth"].setText(f"{dot(depth)} 深度 640×480")
        else:
            self.chips["depth"].setText(f"<span style='color:{WARN}'>●</span> 無深度 · 3D 估計")
        self.chips["fps"].setText(f"{fps:.1f} fps" if fps else "— fps")

    def compose(self, out):
        d = self.display
        frame = out.frame
        if self.mode == "depth":
            img = draw_depth(frame, out.reg_depth)
        elif self.mode == "skeleton":
            img = np.zeros_like(frame)
        elif self.mode == "contour":
            img = np.zeros_like(frame)
            draw_contours(img, out.segmentation.people, thickness=3)
            draw_people(img, out.people, skeleton=False, labels=d["layer_labels"])
            return img
        else:
            img = frame.copy()
            if d["layer_mask"]:
                img = draw_masks(img, out.segmentation.people, d["opacity"])
        draw_people(img, out.people, skeleton=d["layer_skeleton"], labels=d["layer_labels"])
        return img

    def refresh(self):
        if not self.worker:
            return
        result = self.worker.take_output()
        if result is None:
            return
        out, metrics = result
        self.last_image = self.compose(out)
        self.viewport.set_frame(self.last_image)
        if self.view3d is not None and self.view3d.isVisible():
            self.view3d.show_people(out.people, self.display["point_cloud"])
        self.metrics_table.show_people(out.people)
        self.charts.add_frame(out.people, time.monotonic(), out.frame)
        self.inspector.show_metrics(out, metrics)
        self.last_has_depth = metrics.get("has_depth", True)
        self.set_chips(True, metrics["depth"], metrics["fps"], self.last_has_depth)
        if self.stream is not None:
            self.update_stream_info()
        if metrics.get("recorded_frames") is not None and self.recording_dir:
            self.status.setText(f"● 錄製中 · {metrics['recorded_frames']} 幀 · {self.recording_dir}")

    def save_snapshot(self):
        if self.last_image is None:
            self.status.setText("還沒有畫面可以截圖")
            return
        default = PROJECT_ROOT / "captures" / f"astra_{time.strftime('%Y%m%d_%H%M%S')}.png"
        default.parent.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "儲存截圖", str(default), "PNG 圖片 (*.png);;JPEG 圖片 (*.jpg)")
        if path and cv2.imwrite(path, self.last_image):
            self.status.setText(f"已儲存：{path}")

    def closeEvent(self, event):
        if self.discovery:
            self.close_requested = True
            event.ignore()
            return
        if self.worker:
            self.close_requested = True
            self.stop()
            event.ignore()
        else:
            self.set_streaming(False)
            event.accept()
