"""qt-material dark_teal 主題 + 自訂樣式（沿用 Human Mask Studio 的配色）。"""

import sys

from PySide6.QtGui import QFont

ACCENT = "#41dfc8"
BG = "#0c151d"
PANEL = "#14212b"
BORDER = "#263943"
TEXT = "#edf5f7"
MUTED = "#96aab6"
FAINT = "#78939f"
OK, WARN, BAD = "#34d399", "#fbbf24", "#fb7185"

STYLESHEET = f"""
QWidget#root {{ background: {BG}; }}
QLabel {{ background: transparent; margin: 0; padding: 0; }}
QScrollArea#inspectorScroll, QWidget#inspectorBody {{ background: transparent; }}
QLabel#brand {{ font-size: 22px; font-weight: 600; color: {TEXT}; }}
QLabel#eyebrow {{ font-size: 11px; color: {FAINT}; letter-spacing: 1px; }}
QLabel#chip {{ color: {MUTED}; font-size: 12px; padding: 3px 10px; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#section {{ color: {TEXT}; font-size: 13px; font-weight: 600; padding-top: 10px; }}
QLabel#controlLabel, QLabel#metricName, QLabel#caption, QLabel#status {{ color: {MUTED}; font-size: 12px; }}
QLabel#metricValue {{ color: {TEXT}; font-size: 13px; font-weight: 600; }}
QLabel#sliderValue, QLabel#deviceLabel {{ color: {ACCENT}; font-size: 12px; }}
QFrame#inspector {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#personCard {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#cardTitle {{ color: {TEXT}; font-size: 13px; font-weight: 600; }}
QLabel#cardValue {{ color: {TEXT}; font-size: 13px; }}
QLabel#cardNote {{ color: {FAINT}; font-size: 11px; }}
QProgressBar {{ background: {BORDER}; border: none; border-radius: 2px; max-height: 4px; }}
QPushButton {{ text-transform: none; min-height: 24px; border-radius: 7px; }}
QPushButton#primary {{ background: {ACCENT}; color: {BG}; font-weight: 600; }}
QPushButton#mode {{ border-radius: 0; min-width: 64px; }}
QPushButton#mode:checked {{ background: {ACCENT}; color: {BG}; }}
"""


CJK_FONTS = {"darwin": ["PingFang TC", "Heiti TC"],
             "win32": ["Microsoft JhengHei UI", "Microsoft JhengHei"],
             "linux": ["Noto Sans CJK TC", "Noto Sans TC", "WenQuanYi Micro Hei"]}


def ui_font(size=12):
    """依平台挑有繁體中文字形的字型，沒有就交給 Qt 自動替代。"""
    font = QFont()
    font.setFamilies(CJK_FONTS.get(sys.platform, CJK_FONTS["linux"]))
    font.setPointSize(size)
    return font


def apply_theme(app):
    from qt_material import apply_stylesheet

    app.setFont(ui_font(12))
    family = app.font().family()
    apply_stylesheet(app, theme="dark_teal.xml", extra={"font_family": family, "density_scale": "-1"})
    app.setStyleSheet(app.styleSheet() + STYLESHEET)
