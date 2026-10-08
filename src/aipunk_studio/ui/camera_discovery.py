"""在背景執行緒列舉相機，避免介面卡住（取自 Human Mask Studio 的 CameraDiscovery）。"""

from PySide6.QtCore import QThread, Signal


class CameraDiscovery(QThread):
    ready = Signal(object)  # list[Camera]
    failed = Signal(str)

    def __init__(self, discover, parent=None):
        super().__init__(parent)
        self.discover = discover

    def run(self):
        try:
            self.ready.emit(self.discover())
        except Exception as error:
            self.failed.emit(f"無法列出攝影機：{error}")
