"""Qt 背景 worker：把 CaptureRunner 包成 QThread，以 Qt 訊號回報狀態。

沿用 Human Mask Studio 的模式：UI 用計時器呼叫 take_output() 取最新一幀，
worker 永遠只保留最新結果，不會因 UI 忙碌而累積延遲。
"""

import threading

from PySide6.QtCore import QThread, Signal

from .pipeline import Pipeline
from .runner import CaptureRunner


class PipelineWorker(QThread):
    status = Signal(str)
    failed = Signal(str)
    ready = Signal(str)  # 推論裝置說明
    recording = Signal(str, bool)  # (錄製目錄, 是否錄製中)

    def __init__(self, cfg, segmentation=True, parent=None, source_factory=None, pipeline_factory=Pipeline,
                 publisher=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._output = None
        self.runner = CaptureRunner(
            cfg, segmentation, source_factory, pipeline_factory,
            on_status=self.status.emit, on_ready=self.ready.emit,
            on_output=self._store, on_recording=self.recording.emit, publisher=publisher,
        )
        self.cfg = self.runner.cfg

    def _store(self, out, metrics):
        with self._lock:
            self._output = (out, metrics)

    # ---- UI 執行緒呼叫 ----
    def update(self, **changes):
        self.runner.update(**changes)

    def take_output(self):
        """取走最新的 (FrameOutput, metrics)；沒有新結果時回傳 None。"""
        with self._lock:
            out, self._output = self._output, None
            return out

    def stop(self):
        self.runner.stop()

    # ---- worker 執行緒 ----
    def run(self):
        try:
            self.runner.run()
        except Exception as error:
            self.failed.emit(str(error))
