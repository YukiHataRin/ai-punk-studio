"""人物列：每個追蹤 ID 一張卡片，顯示顏色、距離、實測關節比例。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from ...render.colors import track_hex
from ..theme import OK, WARN


class PersonCard(QFrame):
    def __init__(self, track_id):
        super().__init__()
        self.setObjectName("personCard")
        self.setMinimumWidth(150)
        color = track_hex(track_id)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        top = QHBoxLayout()
        title = QLabel(f"<span style='color:{color}'>●</span>&nbsp; ID {track_id}")
        title.setObjectName("cardTitle")
        self.distance = QLabel("—")
        self.distance.setObjectName("cardValue")
        self.distance.setMinimumWidth(72)  # 文字長度會變（≈ 推估值），預留寬度避免被截斷
        self.distance.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.distance)
        layout.addLayout(top)
        self.note = QLabel("")
        self.note.setObjectName("cardNote")
        layout.addWidget(self.note)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)

    def update_person(self, person):
        self.distance.setText(person.distance_text() or "—")
        self.distance.setToolTip("" if person.distance_measured else "沒有量到深度（太近、太遠或被遮擋），此為推估值")
        if person.skeleton is None:
            self.note.setText("僅遮罩（未偵測到骨架）")
            self.bar.setValue(0)
            return
        total = len(person.skeleton.measured)
        measured = int(person.skeleton.measured.sum())
        self.note.setText(f"實測關節 {measured}/{total}" + ("" if measured else "（全部推估）"))
        self.bar.setRange(0, total)
        self.bar.setValue(measured)
        color = OK if measured >= total * 0.6 else WARN
        self.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius: 2px; }}")


class PeoplePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(10)
        self.empty = QLabel("畫面中沒有人")
        self.empty.setObjectName("caption")
        self.layout_.addWidget(self.empty)
        self.layout_.addStretch()
        self.cards: dict[int, PersonCard] = {}

    def show_people(self, people):
        ids = [p.track_id for p in people]
        for tid in list(self.cards):
            if tid not in ids:
                self.cards.pop(tid).deleteLater()
        for i, person in enumerate(people):
            card = self.cards.get(person.track_id)
            if card is None:
                card = PersonCard(person.track_id)
                self.cards[person.track_id] = card
            self.layout_.insertWidget(i, card)  # 依 ID 排序
            card.update_person(person)
        self.empty.setVisible(not people)
