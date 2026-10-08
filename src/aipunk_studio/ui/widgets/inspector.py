"""右側設定面板。所有控制項的變更都經由 changed 訊號送出 (key, value)。

key 分兩類：
- 管線設定（送給 worker）：confidence、tracking、use_mask、point_cloud、min_cutoff_3d、extrinsic_x
- 顯示設定（UI 自己用）：layer_mask、layer_skeleton、layer_labels、opacity
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
)

PIPELINE_KEYS = {"confidence", "tracking", "use_mask", "point_cloud", "min_cutoff_3d", "extrinsic_x"}
POSE_MODELS = (("RTMPose-m（逐人，多人最穩）", "rtmpose"), ("MediaPipe heavy", "heavy"),
               ("MediaPipe full", "full"), ("MediaPipe lite（最快）", "lite"))


class Inspector(QFrame):
    changed = Signal(str, object)
    start_clicked = Signal()
    refresh_clicked = Signal()

    def __init__(self, cfg):
        super().__init__()
        self.setObjectName("inspector")
        self.setMinimumWidth(270)
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(18, 16, 18, 16)
        self.box.setSpacing(6)

        self.section("來源")
        self.camera = QComboBox()
        self.camera.setAccessibleName("攝影機")
        self.camera.currentIndexChanged.connect(self._update_camera_hint)
        self.box.addWidget(self.camera)
        self.refresh = QPushButton("重新整理攝影機")
        self.refresh.clicked.connect(self.refresh_clicked)
        self.box.addWidget(self.refresh)
        self.source = QLabel("尋找攝影機…")
        self.source.setObjectName("controlLabel")
        self.source.setWordWrap(True)
        self.box.addWidget(self.source)
        self.start = QPushButton("開始")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self.start_clicked)
        self.box.addWidget(self.start)
        self.device = QLabel("")
        self.device.setObjectName("deviceLabel")
        self.device.setWordWrap(True)
        self.box.addWidget(self.device)

        self.section("圖層")
        self.check("人體遮罩", "layer_mask", True)
        self.check("2D 骨架", "layer_skeleton", True)
        self.check("ID 與距離", "layer_labels", True)
        self.check("3D 點雲", "point_cloud", cfg["fusion"]["point_cloud"])

        seg = cfg["segmentation"]
        self.section("分割 · YOLO11n-seg")
        self.slider("信心值", "confidence", 0.1, 0.9, seg["confidence"], "{:.2f}")
        self.slider("遮罩透明度", "opacity", 0.1, 0.9, seg["opacity"], "{:.2f}")
        self.check("BoT-SORT 追蹤 ID", "tracking", seg["tracking"])

        self.section("骨架")
        self.model = QComboBox()
        if cfg["pose"].get("backend", "rtmpose") == "rtmpose":
            current = "rtmpose"
        else:
            current = cfg["pose"]["model"].rsplit("_", 1)[-1].removesuffix(".task")
        for text, key in POSE_MODELS:
            self.model.addItem(text, key)
        keys = [k for _, k in POSE_MODELS]
        self.model.setCurrentIndex(keys.index(current) if current in keys else 0)
        self.label("模型（停止時可切換）")
        self.box.addWidget(self.model)
        self.slider("平滑（越小越穩）", "min_cutoff_3d", 0.1, 3.0, cfg["smoothing"]["min_cutoff_3d"], "{:.1f} Hz")

        self.section("深度")
        self.check("只取自己遮罩內的深度", "use_mask", cfg["fusion"]["use_mask"])
        self.slider("對齊 x 平移", "extrinsic_x", -0.06, 0.02, cfg["extrinsics"]["translation"][0], "{:+.0f} mm", 1000)

        self.section("串流（WebSocket）")
        self.check("啟用 WebSocket 串流", "stream_enabled", cfg["stream"]["enabled"])
        self.stream_info = QLabel("未啟用")
        self.stream_info.setObjectName("controlLabel")
        self.stream_info.setWordWrap(True)
        self.stream_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.box.addWidget(self.stream_info)

        self.section("錄製")
        self.check("包含 RGB-D 影像（可回放、檔案較大）", "record_images", True)
        self.label("只勾骨架時僅寫 skeleton.jsonl")

        self.section("效能")
        self.metrics = {}
        for key, text in (("fps", "處理 FPS"), ("segmentation", "YOLO 分割"), ("pose", "MediaPipe 骨架"),
                          ("fusion", "融合 + 3D"), ("people", "人數")):
            row = QHBoxLayout()
            name = QLabel(text)
            name.setObjectName("metricName")
            value = QLabel("—")
            value.setObjectName("metricValue")
            row.addWidget(name)
            row.addStretch()
            row.addWidget(value)
            self.box.addLayout(row)
            self.metrics[key] = value
        self.box.addStretch()

    # ---- 建立控制項 ----
    def section(self, text):
        label = QLabel(text)
        label.setObjectName("section")
        self.box.addWidget(label)

    def label(self, text):
        label = QLabel(text)
        label.setObjectName("controlLabel")
        self.box.addWidget(label)

    def check(self, text, key, value):
        box = QCheckBox(text)
        box.setChecked(bool(value))
        box.toggled.connect(lambda v: self.changed.emit(key, v))
        self.box.addWidget(box)
        setattr(self, f"{key}_box", box)
        return box

    def slider(self, text, key, low, high, value, fmt, display_scale=1):
        """浮點滑桿：內部用 0–1000 整數，顯示時乘上 display_scale 再套 fmt。"""
        row = QHBoxLayout()
        name = QLabel(text)
        name.setObjectName("controlLabel")
        out = QLabel()
        out.setObjectName("sliderValue")
        row.addWidget(name)
        row.addStretch()
        row.addWidget(out)
        self.box.addLayout(row)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 1000)
        to_value = lambda i: low + (high - low) * i / 1000  # noqa: E731
        s.setValue(round((value - low) / (high - low) * 1000))
        out.setText(fmt.format(value * display_scale))

        def on_change(i):
            v = to_value(i)
            out.setText(fmt.format(v * display_scale))
            self.changed.emit(key, v)

        s.valueChanged.connect(on_change)
        self.box.addWidget(s)
        setattr(self, f"{key}_slider", s)
        return s

    # ---- 攝影機 ----
    def set_cameras(self, cameras, preferred=None):
        """cameras：list[Camera]。preferred：要選取的 Camera；沒指定時優先選 Astra Pro。"""
        self.camera.blockSignals(True)
        self.camera.clear()
        for cam in cameras:
            self.camera.addItem(f"{cam.name}（{'RGB-D' if cam.is_astra else '僅 RGB'}）", cam)
        if cameras:
            target = preferred or next((c for c in cameras if c.is_astra), cameras[0])
            self.camera.setCurrentIndex(next((i for i, c in enumerate(cameras) if c.uid == target.uid), 0))
        self.camera.blockSignals(False)
        self._update_camera_hint()

    def selected_camera(self):
        return self.camera.currentData()

    def _update_camera_hint(self, *_):
        cam = self.selected_camera()
        if cam is None:
            self.source.setText("沒有偵測到攝影機")
        elif cam.is_astra:
            self.source.setText("RGB + 深度：3D 為實測")
        else:
            self.source.setText("沒有深度：3D 由身體尺寸估計（約略值）")

    def set_playback(self, name):
        self.camera.setEnabled(False)
        self.refresh.setEnabled(False)
        self.source.setText(f"回放：{name}")

    # ---- 狀態 ----
    def set_running(self, running):
        self.start.setText("停止" if running else "開始")
        self.model.setEnabled(not running)
        self.camera.setEnabled(not running)
        self.refresh.setEnabled(not running)

    def pose_model(self):
        return self.model.currentData()

    def show_metrics(self, out, metrics):
        t = out.timings
        self.metrics["fps"].setText(f"{metrics['fps']:.1f}")
        self.metrics["segmentation"].setText(f"{t['segmentation']:.0f} ms")
        self.metrics["pose"].setText(f"{t['pose']:.0f} ms")
        self.metrics["fusion"].setText(f"{t['fusion']:.1f} ms")
        self.metrics["people"].setText(str(len(out.people)))
