"""總覽頁的指標表格：每位舞者一列，欄位為 ID、距離、實測關節與九項舞蹈指標。"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QStyle, QStyledItemDelegate, QTableWidget, QTableWidgetItem,
)

from ...core.dance_metrics import METRIC_JOINTS, METRICS, metric_joint_coverage
from ...render.colors import track_hex
from ..theme import FAINT, TEXT

BASE_COLUMNS = (("ID", "追蹤編號（顏色與畫面一致）"), ("距離", "髖部中心到相機的距離；≈ 為推估值"),
                (f"實測關節\n（共 {len(METRIC_JOINTS)}）",
                 "指標用到的 13 個關節（鼻、肩、肘、腕、髖、膝、踝）中，由深度實際量到的數量；"
                 "0 表示這個人的指標完全來自推估骨架"))


class ColoredTextDelegate(QStyledItemDelegate):
    """qt-material 的樣式表把表格文字固定成白色（QTableView::item { color }），會蓋掉每格的前景色；
    這裡自己畫文字，讓 ID 顏色與推估值的灰色生效。"""

    def paint(self, painter, option, index):
        self.initStyleOption(option, index)
        text, option.text = option.text, ""
        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        painter.save()
        painter.setPen(brush.color() if brush is not None else QColor(TEXT))
        painter.setFont(option.font)
        painter.drawText(option.rect, int(Qt.AlignmentFlag.AlignCenter), text)
        painter.restore()


def format_value(value):
    if value is None:
        return "—"
    a = abs(value)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{value:.3g}"
    if a >= 10:
        return f"{value:.1f}"
    if a >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


class MetricsTable(QTableWidget):
    def __init__(self):
        super().__init__(0, len(BASE_COLUMNS) + len(METRICS))
        headers = [name for name, _ in BASE_COLUMNS] + [f"{m.name}\n{m.unit}" for m in METRICS]
        self.setHorizontalHeaderLabels(headers)
        for col, (_, tip) in enumerate(BASE_COLUMNS):
            self.horizontalHeaderItem(col).setToolTip(tip)
        for i, m in enumerate(METRICS):
            self.horizontalHeaderItem(len(BASE_COLUMNS) + i).setToolTip(f"{m.name}（{m.name_en}）：{m.description}")
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setMinimumSectionSize(54)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumHeight(150)
        self.setMaximumHeight(220)
        self.setItemDelegate(ColoredTextDelegate(self))

    def _set(self, row, col, text, color=TEXT, tooltip=""):
        item = self.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, col, item)
        item.setText(text)
        item.setForeground(QBrush(QColor(color)))
        item.setToolTip(tooltip)

    def show_people(self, people):
        self.setRowCount(len(people))
        for row, p in enumerate(people):
            sk = p.skeleton
            estimated = not p.metrics_measured  # 指標用的 13 個關節都沒量到深度：指標全部來自推估骨架
            value_color = FAINT if estimated else TEXT
            self._set(row, 0, f"● {p.track_id}", track_hex(p.track_id))
            self._set(row, 1, p.distance_text() or "—", FAINT if not p.distance_measured else TEXT)
            if sk is None:
                self._set(row, 2, "僅遮罩", FAINT, "沒有偵測到骨架，無法計算指標")
            else:
                measured, total = metric_joint_coverage(sk)
                self._set(row, 2, f"{measured}/{total}", value_color)
            tip = "指標用到的關節都沒量到深度，指標由推估骨架計算" if estimated else ""
            for i, m in enumerate(METRICS):
                value = None if p.metrics is None else p.metrics.get(m.key)
                self._set(row, len(BASE_COLUMNS) + i, format_value(value), value_color, tip)
