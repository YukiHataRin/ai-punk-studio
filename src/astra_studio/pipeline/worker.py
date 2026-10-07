"""Qt 背景 worker：開相機、載模型、逐幀跑 Pipeline，把最新結果放進信箱給 UI 取。

沿用 Human Mask Studio 的模式：UI 用計時器呼叫 take_output() 取最新一幀，
worker 永遠只保留最新結果，不會因 UI 忙碌而累積延遲。
設定變更透過 update() 丟進待套用清單，由 worker 執行緒在下一幀開始前套用。
"""

import copy
import threading
import time

from PySide6.QtCore import QThread, Signal

from ..io.recorder import SessionRecorder
from .pipeline import Pipeline


def find_astra_rgb_index(default):
    """在相機清單中找 Astra Pro 的 RGB 鏡頭；找不到就用設定檔的編號。"""
    try:
        from ..sensors.discovery import discover_cameras
        for camera in discover_cameras():
            if camera.is_astra:
                return camera.index
    except Exception:
        pass
    return default


class PipelineWorker(QThread):
    status = Signal(str)
    failed = Signal(str)
    ready = Signal(str)  # 推論裝置說明
    recording = Signal(str, bool)  # (錄製目錄, 是否錄製中)

    def __init__(self, cfg, segmentation=True, parent=None, source_factory=None, pipeline_factory=Pipeline):
        super().__init__(parent)
        self.cfg = copy.deepcopy(cfg)  # worker 執行緒專用，UI 改設定一律走 update()
        self.segmentation = segmentation
        self.source_factory = source_factory
        self.pipeline_factory = pipeline_factory
        self._lock = threading.Lock()
        self._pending = {}
        self._output = None
        self._stop = threading.Event()
        self.recorder = None

    # ---- UI 執行緒呼叫 ----
    def update(self, **changes):
        with self._lock:
            self._pending.update(changes)

    def take_output(self):
        """取走最新的 (FrameOutput, metrics)；沒有新結果時回傳 None。"""
        with self._lock:
            out, self._output = self._output, None
            return out

    def stop(self):
        self._stop.set()

    # ---- worker 執行緒 ----
    def _apply_pending(self, pipeline):
        with self._lock:
            changes, self._pending = self._pending, {}
        cfg = self.cfg
        for key, value in changes.items():
            if key == "confidence":
                cfg["segmentation"]["confidence"] = value
            elif key == "tracking":
                cfg["segmentation"]["tracking"] = value
                pipeline.reset_tracking()
            elif key == "use_mask":
                cfg["fusion"]["use_mask"] = value
            elif key == "point_cloud":
                cfg["fusion"]["point_cloud"] = value
            elif key == "min_cutoff_3d":
                pipeline.smoother.set_params(min_cutoff_3d=value)
            elif key == "extrinsic_x":
                pipeline.registration.t[0] = value
            elif key == "record":
                self._set_recording(value)

    def _set_recording(self, options):
        """options：None 停止；dict(with_images=bool) 開始。"""
        if self.recorder is not None:
            path = self.recorder.close()
            self.recorder = None
            self.recording.emit(str(path), False)
        if options:
            self.recorder = SessionRecorder(self.cfg, with_images=options.get("with_images", True))
            self.recording.emit(str(self.recorder.dir), True)

    def _open_source(self):
        """依 cfg["source"] 開啟來源：astra = RGB + 深度；rgb = 只有 RGB（3D 改用估計）。
        沒有指定時自動找 Astra Pro 的 RGB 鏡頭。Astra 的深度開不起來時退回只用 RGB。"""
        if self.source_factory is not None:
            return self.source_factory(self.cfg).start()
        from ..sensors.astra import AstraSource, DepthUnavailable
        from ..sensors.webcam import WebcamSource

        src = self.cfg.get("source") or {"kind": "astra", "index": find_astra_rgb_index(self.cfg["rgb"]["index"])}
        self.cfg["rgb"]["index"] = src["index"]
        if src["kind"] != "astra":
            self.status.emit(f"開啟 {src.get('name', '攝影機')}（僅 RGB）…")
            return WebcamSource(self.cfg).start()
        self.status.emit("開啟 Astra Pro…")
        try:
            return AstraSource(self.cfg).start()
        except DepthUnavailable as error:
            self.status.emit(f"深度相機無法開啟（{error}），改用僅 RGB，3D 為估計值")
            return WebcamSource(self.cfg).start()

    def run(self):
        source = pipeline = None
        try:
            self.status.emit("載入模型中…")
            pipeline = self.pipeline_factory(self.cfg, segmentation=self.segmentation)
            device = pipeline.segmenter.device.label if pipeline.segmenter else "—"
            self.ready.emit(f"YOLO：{device} · {pipeline.pose_label}")
            if self._stop.is_set():
                return

            source = self._open_source()
            has_depth = getattr(source, "has_depth", True)
            self.status.emit("等待影像…" if has_depth else "等待影像…（沒有深度，3D 為估計值）")

            last, frames, t0, fps, live = 0.0, 0, time.perf_counter(), 0.0, False
            while not self._stop.is_set():
                if source.error:
                    raise RuntimeError(source.error)
                if getattr(source, "finished", False):
                    self.status.emit("播放完畢")
                    break
                frame, stamp, depth = source.latest()
                if frame is None or stamp == last:
                    self._stop.wait(0.002)
                    continue
                last = stamp
                self._apply_pending(pipeline)
                out = pipeline.process(frame, depth, stamp)
                if self.recorder is not None:
                    self.recorder.write(frame, depth, out, stamp)
                frames += 1
                now = time.perf_counter()
                if now - t0 >= 1.0:
                    fps, frames, t0 = frames / (now - t0), 0, now
                if not live:
                    live = True
                    self.status.emit("即時中")
                rec = None if self.recorder is None else self.recorder.index
                with self._lock:
                    self._output = (out, {"fps": fps, "depth": depth is not None, "has_depth": has_depth,
                                          "recorded_frames": rec})
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if self.recorder is not None:
                try:
                    self._set_recording(None)
                except Exception as error:
                    self.failed.emit(str(error))
            if source:
                source.stop()
            if pipeline:
                pipeline.close()
