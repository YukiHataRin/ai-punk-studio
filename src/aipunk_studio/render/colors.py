"""每個追蹤 ID 一個顏色；遮罩、骨架、標籤、人物卡片共用（色票取自 Human Mask Studio）。"""

TRACK_COLORS = ("#22D3EE", "#FB7185", "#A78BFA", "#FBBF24", "#34D399", "#F97316", "#60A5FA", "#E879F9")
UNTRACKED = "#20D7F5"


def hex_to_bgr(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))


def track_hex(track_id):
    return UNTRACKED if track_id is None else TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]


def track_bgr(track_id):
    return hex_to_bgr(track_hex(track_id))
