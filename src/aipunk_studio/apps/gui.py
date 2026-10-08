"""Qt 介面入口：建立 QApplication、套主題、開主視窗。"""

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ..ui.main_window import MainWindow
from ..ui.theme import apply_theme


def run(cfg, segmentation=True, play=None, camera=None, view=None, tab=None, start=False,
        screenshot=None, screenshot_delay=12):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AI Punk Studio")
    apply_theme(app)
    window = MainWindow(cfg, segmentation=segmentation, playback=Path(play) if play else None,
                        initial_camera=camera)
    window.show()
    if view:
        window.select_mode(view)
    if tab == "metrics":
        window.tabs.setCurrentWidget(window.charts)
    if start:
        QTimer.singleShot(0, window.start)
    if screenshot:  # 開發用：延遲後存視窗截圖並結束
        def capture():
            window.grab().save(screenshot)
            window.close()
            QTimer.singleShot(4000, app.quit)
        QTimer.singleShot(int(screenshot_delay * 1000), capture)
    return app.exec()
