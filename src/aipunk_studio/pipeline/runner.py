"""擷取迴圈（不依賴 Qt）：開來源、載模型、逐幀跑 Pipeline，透過回呼回報結果。

Qt 介面（ui/worker.py）與 headless 伺服器（apps/headless.py）共用這個迴圈。
設定變更透過 update() 丟進待套用清單，由擷取執行緒在下一幀開始前套用。
"""

import copy
import threading
import time

from ..io.recorder import SessionRecorder
from ..sensors.discovery import find_astra_rgb_index
from .pipeline import Pipeline


def _noop(*args, **kwargs):
    pass


class CaptureRunner:
    def __init__(self, cfg, segmentation=True, source_factory=None, pipeline_factory=Pipeline,
                 on_status=_noop, on_ready=_noop, on_output=_noop, on_recording=_noop, publisher=None):
        """回呼都在擷取執行緒呼叫：
        on_status(str)、on_ready(裝置說明)、on_output(FrameOutput, metrics)、on_recording(目錄, 是否錄製中)。
        publisher：有 start_session(cfg, has_depth) 與 publish(out, metrics) 的物件（例如 StreamServer），可省略。"""
        self.cfg = copy.deepcopy(cfg)  # 擷取執行緒專用，外部改設定一律走 update()
        self.segmentation = segmentation
        self.source_factory = source_factory
        self.pipeline_factory = pipeline_factory
        self.on_status, self.on_ready, self.on_output, self.on_recording = on_status, on_ready, on_output, on_recording
        self.publisher = publisher
        self.recorder = None
        self._lock = threading.Lock()
        self._pending = {}
        self._stop = threading.Event()

    # ---- 其他執行緒呼叫 ----
    def update(self, **changes):
        with self._lock:
            self._pending.update(changes)

    def stop(self):
        self._stop.set()

    @property
    def stopping(self):
        return self._stop.is_set()

    # ---- 擷取執行緒 ----
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
            self.on_recording(str(path), False)
        if options:
            self.recorder = SessionRecorder(self.cfg, with_images=options.get("with_images", True))
            self.on_recording(str(self.recorder.dir), True)

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
            self.on_status(f"開啟 {src.get('name', '攝影機')}（僅 RGB）…")
            return WebcamSource(self.cfg).start()
        self.on_status("開啟 Astra Pro…")
        try:
            return AstraSource(self.cfg).start()
        except DepthUnavailable as error:
            self.on_status(f"深度相機無法開啟（{error}），改用僅 RGB，3D 為估計值")
            return WebcamSource(self.cfg).start()

    def run(self):
        """執行到 stop() 或來源結束；發生錯誤時拋出例外（呼叫端負責回報）。"""
        source = pipeline = None
        try:
            self.on_status("載入模型中…")
            pipeline = self.pipeline_factory(self.cfg, segmentation=self.segmentation)
            device = pipeline.segmenter.device.label if pipeline.segmenter else "—"
            self.on_ready(f"YOLO：{device} · {pipeline.pose_label}")
            if self._stop.is_set():
                return

            source = self._open_source()
            has_depth = getattr(source, "has_depth", True)
            if self.publisher is not None:
                self.publisher.start_session(self.cfg, has_depth)
            self.on_status("等待影像…" if has_depth else "等待影像…（沒有深度，3D 為估計值）")

            last, frames, t0, fps, live = 0.0, 0, time.perf_counter(), 0.0, False
            while not self._stop.is_set():
                if source.error:
                    raise RuntimeError(source.error)
                if getattr(source, "finished", False):
                    self.on_status("播放完畢")
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
                    self.on_status("即時中")
                metrics = {"fps": fps, "depth": depth is not None, "has_depth": has_depth,
                           "recorded_frames": None if self.recorder is None else self.recorder.index}
                if self.publisher is not None:
                    self.publisher.publish(out, metrics)
                self.on_output(out, metrics)
        finally:
            if self.recorder is not None:
                self._set_recording(None)
            if source:
                source.stop()
            if pipeline:
                pipeline.close()
