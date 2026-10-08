"""2D 主畫面：等比例縮放顯示 BGR 影像（取自 Human Mask Studio 的 Preview）。"""

import cv2
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from ..theme import BORDER, MUTED, TEXT


class Viewport(QWidget):
    def __init__(self, title="看見每一個人", hint="選擇攝影機後按「開始」"):
        super().__init__()
        self.image = QImage()
        self.title, self.hint = title, hint
        self.setMinimumSize(480, 300)

    def set_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def clear(self):
        self.image = QImage()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(BORDER))
        p.setBrush(QColor("#101b23"))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 14, 14)
        if not self.image.isNull():
            size = self.image.size().scaled(self.size() - QSize(16, 16), Qt.AspectRatioMode.KeepAspectRatio)
            x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
            p.drawImage(QRect(x, y, size.width(), size.height()), self.image)
            return
        family = QApplication.font().family()
        p.setPen(QColor(TEXT))
        p.setFont(QFont(family, 22, QFont.Weight.DemiBold))
        p.drawText(self.rect().adjusted(0, -30, 0, -30), Qt.AlignmentFlag.AlignCenter, self.title)
        p.setPen(QColor(MUTED))
        p.setFont(QFont(family, 12))
        p.drawText(self.rect().adjusted(0, 36, 0, 36), Qt.AlignmentFlag.AlignCenter, self.hint)
