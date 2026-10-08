"""產生演算法架構圖 docs/figures/architecture.svg（Nature 期刊風格，全向量、文字可編輯）。

骨架形狀取自 MediaPipe 對範例影像的實際輸出（docs/figures/pose_sample.json）；
One Euro 曲線由 aipunk_studio.core.filters.OneEuroFilter 實際計算；
時序圖數值為本機實測（TIMINGS：Apple M5、1280×720，pipeline.process 的 timings 中位數）。

    python tools/make_architecture_figure.py
"""

import json
from pathlib import Path

import numpy as np

from aipunk_studio.core.filters import OneEuroFilter
from aipunk_studio.render.colors import TRACK_COLORS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures" / "architecture.svg"
POSE = json.loads((ROOT / "docs" / "figures" / "pose_sample.json").read_text())

# 實測每幀耗時（ms，中位數）：pipeline.process 的 timings + DepthToRgb 單獨量測
TIMINGS = {"segmentation": 22.5, "pose": 4.2, "d2c": 1.9, "fusion": 1.3, "total": 25.3, "total_3p": 35.0}

W, H = 1800, 1030
FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang TC', 'Noto Sans TC', sans-serif"
INK, GREY, MUTE, RULE = "#1f1f1f", "#5f6368", "#9aa0a6", "#d5d9de"
C = {"rgb": ("#2f6db5", "#e8f0fa"), "depth": ("#d9822b", "#fcf1e4"),
     "fuse": ("#2f8f5b", "#e6f4ec"), "out": ("#7a5aa6", "#f1ecf7"), "track": ("#3f4a54", "#f2f4f6")}
ID1, ID2 = TRACK_COLORS[0], TRACK_COLORS[1]  # 與程式中 ID 1、ID 2 的顏色一致
BODY_CONN = [(a, b) for a, b in POSE["conn"] if a >= 11 and b >= 11]
VIS = np.array(POSE["vis"])

svg = []


def add(s):
    svg.append(s)


def text(x, y, s, size=14, color=INK, weight="normal", anchor="start", italic=False):
    style = ' font-style="italic"' if italic else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-weight="{weight}" '
        f'text-anchor="{anchor}"{style}>{s}</text>')


def lines(x, y, rows, size=13, color=INK, gap=1.5, **kw):
    for i, r in enumerate(rows):
        text(x, y + i * size * gap, r, size, color, **kw)


def rect(x, y, w, h, fill="none", stroke="none", sw=1, r=8, extra=""):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def line(x1, y1, x2, y2, color, sw=1.0, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
        f'stroke-width="{sw}" stroke-linecap="round"{da}/>')


def arrow(pts, color, sw=2.2, dash=None):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} stroke-linejoin="round" '
        f'marker-end="url(#ah-{color.strip("#")})"/>')


def panel(x, y, letter, title):
    text(x, y, letter, 25, INK, "bold")
    text(x + 26, y - 2, title, 16.5, INK, "bold")


def sub(s):
    return f"<tspan baseline-shift='sub' font-size='9'>{s}</tspan>"


def it(s):
    return f"<tspan font-style='italic'>{s}</tspan>"


# ---------------------------------------------------------------- 人形
def body(cx, cy, h, mirror=False):
    """以髖部中心 (cx, cy) 為基準、身高約 h 的 33 點人形（取自實際 MediaPipe 輸出）。"""
    n = np.array(POSE["norm"], np.float64)
    n[:, 0] *= POSE["aspect"]
    n -= n[[23, 24]].mean(0)
    span = n[11:, 1].max() - n[11:, 1].min() + 0.12
    if mirror:
        n[:, 0] *= -1
    return np.array([cx, cy]) + n * (h / span)


def silhouette(P, color, width, opacity=1.0):
    """粗線 + 頭部圓形 + 軀幹多邊形當作人體剪影（也用來畫遮罩）。"""
    op = f' opacity="{opacity}"' if opacity < 1 else ""
    add(f"<g{op}>")
    for a, b in BODY_CONN:
        if min(VIS[a], VIS[b]) >= 0.3:
            line(*P[a], *P[b], color, width)
    head = P[0] if VIS[0] >= 0.3 else P[[11, 12]].mean(0) - [0, width]
    add(f'<circle cx="{head[0]:.1f}" cy="{head[1]:.1f}" r="{width * 0.85:.1f}" fill="{color}"/>')
    add(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in P[[11, 12, 24, 23]])}" fill="{color}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
    add("</g>")


def outline(P, color, width, bg):
    silhouette(P, color, width)
    silhouette(P, bg, width - 3)


def skeleton(P, color, sw=1.6, dots=True):
    for a, b in BODY_CONN:
        if min(VIS[a], VIS[b]) >= 0.3:
            line(*P[a], *P[b], color, sw)
    if dots:
        for j in range(11, 33):
            if VIS[j] >= 0.3:
                add(f'<circle cx="{P[j,0]:.1f}" cy="{P[j,1]:.1f}" r="{sw * 1.1:.1f}" fill="{color}"/>')


def tag(x, y, label, color, size=10):
    w = len(label) * size * 0.62 + 8
    rect(x, y - size - 3, w, size + 6, color, r=2)
    text(x + 4, y, label, size, "#0b1b22", "bold")


# ---------------------------------------------------------------- 開頭
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">')
add("<defs>")
for col in [INK, GREY, MUTE] + [v[0] for v in C.values()]:
    add(f'<marker id="ah-{col.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add('<linearGradient id="g-depth" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7a0403"/>'
    '<stop offset="1" stop-color="#b8250a"/></linearGradient>')
add('<pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    f'<line x1="0" y1="0" x2="0" y2="5" stroke="{ID2}" stroke-width="2"/></pattern>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

rgb_col, depth_col, fuse_col, out_col, trk_col = (C[k][0] for k in ("rgb", "depth", "fuse", "out", "track"))
TOP = 70

# ================================================================ a 感測器
AX = 40
panel(AX, TOP - 18, "a", "感測器與擷取")
add('<g id="asset-camera">')  # 可替換為插圖素材
cx0, cy0 = AX + 15, TOP + 215
rect(cx0 + 80, cy0 + 40, 40, 58, "#3a3a3a", r=0)
rect(cx0 + 45, cy0 + 96, 110, 12, "#2b2b2b", r=4)
rect(cx0, cy0, 200, 44, "#2b2b2b", r=12)
rect(cx0 + 4, cy0 + 4, 192, 12, "#ffffff", r=6, extra='opacity="0.08"')
for lx, r in [(cx0 + 40, 9), (cx0 + 100, 11), (cx0 + 160, 9)]:
    add(f'<circle cx="{lx}" cy="{cy0 + 22}" r="{r}" fill="#111" stroke="#666" stroke-width="1.5"/>')
    add(f'<circle cx="{lx - 3}" cy="{cy0 + 19}" r="{r * 0.3:.1f}" fill="#8fb4e0" opacity="0.7"/>')
add("</g>")
text(AX + 115, cy0 + 138, "Orbbec Astra Pro", 15.5, INK, "bold", "middle")
text(AX + 115, cy0 + 157, "USB 2.0", 12.5, GREY, anchor="middle")
lines(AX, TOP + 40, ["RGB（UVC）", "1280×720 · 30 fps"], 13.5, rgb_col, weight="bold")
text(AX, TOP + 82, "OpenCV / AVFoundation", 12, GREY)
lines(AX, TOP + 420, ["深度（OpenNI 協定）", "640×480 · 30 fps"], 13.5, depth_col, weight="bold")
text(AX, TOP + 462, "pyorbbecsdk v1", 12, GREY)
lines(AX, TOP + 495, ["兩路各一條擷取執行緒，", "主迴圈只取最新影格"], 12, GREY)

# ================================================================ b 分割 + ID
BX, BW = 330, 410
BY, BH = TOP, 160
panel(BX, BY - 18, "b", "人體分割與 ID 追蹤")
rect(BX, BY, BW, BH, C["rgb"][1], r=10)
tx, tw, th = BX + 14, 156, 104
ty = BY + 14
add('<g id="asset-seg-frame">')
rect(tx, ty, tw, th, "#dbe7f5", "#b9cde6", r=4)
silhouette(body(tx + 52, ty + 62, 82), ID1, 9, 0.75)
silhouette(body(tx + 108, ty + 66, 74, mirror=True), ID2, 9, 0.75)
add("</g>")
tag(tx + 24, ty + 14, "ID 1", ID1)
tag(tx + 96, ty + 22, "ID 2", ID2)
text(BX + 186, BY + 30, "YOLO11n-seg（GPU · MPS）", 14, INK, "bold")
lines(BX + 186, BY + 56, ["BoT-SORT + ReID：跨幀維持 ID", "retina_masks：原圖解析度遮罩", "人交錯、短暫遮擋仍保留 ID"], 12.5, INK)
text(BX + 14, BY + BH - 10, "輸出／人：遮罩 (H×W)、框、track ID", 12, rgb_col)

# ================================================================ c 骨架
CY = BY + BH + 50
panel(BX, CY - 18, "c", "逐人骨架估計（由上而下）")
rect(BX, CY, BW, BH, C["rgb"][1], r=10)
add('<g id="asset-pose-frame">')
rect(tx, CY + 14, tw, th, "#dbe7f5", "#b9cde6", r=4)
for cx, cy, hh, mir, col in ((tx + 52, CY + 76, 82, False, ID1), (tx + 108, CY + 80, 74, True, ID2)):
    P = body(cx, cy, hh, mir)
    x0, y0 = P[11:].min(0) - 5
    x1, y1 = P[11:].max(0) + 5
    top = min(y0, P[0, 1] - 10)
    rect(x0, top, x1 - x0, y1 - top, "none", col, 1.4, r=2, extra='stroke-dasharray="3 2"')
    skeleton(P, rgb_col, 1.5)
add("</g>")
text(BX + 186, CY + 30, "RTMPose-m（CoreML）", 14, INK, "bold")
lines(BX + 186, CY + 56, ["每個人框各估一次，約 4 ms/人", "用上一幀人框，與 b 平行執行", "骨架直接屬於該框的 track ID"], 12.5, INK)
text(BX + 14, CY + BH - 10, "輸出／人：Halpe26 26 關節 (u, v)、信心值（含腳趾、腳跟）", 12, rgb_col)

# ================================================================ d D2C
DY = CY + BH + 50
DH = 132
panel(BX, DY - 18, "d", "深度對齊（D2C）")
rect(BX, DY, BW, DH, C["depth"][1], r=10)
gx, gy = BX + 34, DY + 96
add(f'<polygon points="{gx},{gy} {gx - 12},{gy + 19} {gx + 12},{gy + 19}" fill="{depth_col}"/>')
add(f'<polygon points="{gx + 54},{gy} {gx + 42},{gy + 19} {gx + 66},{gy + 19}" fill="{rgb_col}"/>')
px_, py_ = gx + 34, DY + 22
add(f'<circle cx="{px_}" cy="{py_}" r="4.5" fill="{INK}"/>')
text(px_ + 9, py_ + 2, "P", 13, INK, "bold", italic=True)
line(gx, gy, px_, py_, depth_col, 1.5)
line(gx + 54, gy, px_, py_, rgb_col, 1.5, "4 3")
lines(BX + 124, DY + 28, [
    f"深度像素 → 3D：{it('P')} = {it('z')}·K{sub('D')}<tspan baseline-shift='super' font-size='9'>−1</tspan>[{it('u')}, {it('v')}, 1]ᵀ",
    f"轉到 RGB：{it('P′')} = R{it('P')} + {it('t')}，投影 π(K{sub('C')}{it('P′')})",
    "z-buffer 由遠到近 → 640×360 對齊深度",
    f"{it('t')} = (−25, 0, 0) mm（依原廠規格估計）",
], 12, INK, gap=1.6)

# ================================================================ e 融合
EX, EW = 800, 380
panel(EX, TOP - 18, "e", "融合與 3D 提升")
rect(EX, TOP, EW, 552, C["fuse"][1], r=10)

text(EX + 16, TOP + 26, "① 骨架歸屬", 14, INK, "bold")
fx, fy = EX + 16, TOP + 38
rect(fx, fy, 120, 96, "#ffffff", RULE, r=4)
P1 = body(fx + 60, fy + 54, 76)
silhouette(P1, ID1, 9, 0.45)
skeleton(P1, INK, 1.3)
lines(EX + 150, TOP + 60, ["RTMPose：骨架來自該人框，", "依 track ID 對到這一幀的遮罩", "（MediaPipe 模式：依關節落在", "遮罩內的比例配對）"], 12.5, INK)

text(EX + 16, TOP + 168, "② 只取「自己遮罩內」的深度", 14, INK, "bold")
ox, oy = EX + 16, TOP + 180
rect(ox, oy, 160, 130, "url(#g-depth)", r=4)
back = body(ox + 62, oy + 72, 100)
front = body(ox + 106, oy + 80, 104, mirror=True)
silhouette(back, ID1, 11, 0.9)
silhouette(front, ID2, 11, 0.9)
wj = back[14]  # 後方那人的右肘，被前方的人擋到一部分
rect(wj[0] - 11, wj[1] - 11, 22, 22, "none", "#ffffff", 1.6, r=1)
tag(ox + 6, oy + 16, "ID 1 後方", ID1, 9)
tag(ox + 98, oy + 124, "ID 2 前方", ID2, 9)
lx = EX + 190
rect(lx, TOP + 188, 14, 14, ID1, r=2)
text(lx + 20, TOP + 200, "自己遮罩內 → 採用", 12, INK)
rect(lx, TOP + 212, 14, 14, "url(#hatch)", ID2, 1, r=2)
text(lx + 20, TOP + 224, "別人的遮罩 → 忽略", 12, INK)
lines(lx, TOP + 252, [
    f"視窗 ∩ 自己遮罩 ∩ |{it('z')} − {it('z')}{sub('ref')}| &lt; 0.6 m",
    "取中位數 + 5 cm（表面 → 關節）",
    "視窗內沒有自己的遮罩 → 視為",
    "被遮擋，改用推估值",
], 12, INK, gap=1.55)

text(EX + 16, TOP + 344, "③ 每人輸出（RGB 相機座標系，公尺）", 14, INK, "bold")
rows = [(fuse_col, "有骨架", "26 關節 3D，標記實測／推估"),
        (ID1, "只有遮罩", "遮罩深度中位數 → 3D 位置"),
        (MUTE, "都量不到", "距離前加 ≈，表示推估值")]
for i, (col, k, v) in enumerate(rows):
    y = TOP + 372 + i * 30
    add(f'<circle cx="{EX + 24}" cy="{y - 4}" r="5" fill="{col}"/>')
    text(EX + 36, y, k, 12.5, INK, "bold")
    text(EX + 106, y, v, 12.5, INK)
lines(EX + 16, TOP + 500, ["推估：2D 位置照用，深度取這個人", "實測關節的中位數（人體所在深度平面）"], 12, GREY, gap=1.5)

# ================================================================ f ID 與平滑
FX, FW = 1225, 250
panel(FX, TOP - 18, "f", "ID 與時間平滑")
rect(FX, TOP, FW, 552, C["track"][1], r=10)
text(FX + 16, TOP + 26, "兩人交錯，ID 不交換", 13.5, INK, "bold")
frames = 5
gx0, gw = FX + 34, FW - 60
for k in range(frames):
    x = gx0 + k * gw / (frames - 1)
    line(x, TOP + 44, x, TOP + 124, RULE, 1)
    text(x, TOP + 140, f"t{sub(str(k + 1))}", 11, GREY, anchor="middle")
ya = [TOP + 56, TOP + 70, TOP + 84, TOP + 98, TOP + 112]
for col, ys in ((ID1, ya), (ID2, ya[::-1])):
    pts = [(gx0 + k * gw / (frames - 1), ys[k]) for k in range(frames)]
    add(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
    for x, y in pts:
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}"/>')
tag(FX + 8, TOP + 60, "1", ID1, 9)
tag(FX + 8, TOP + 117, "2", ID2, 9)
lines(FX + 16, TOP + 166, ["ID 來自 BoT-SORT；關閉追蹤時", "改用骨架位置配對"], 12, INK)

text(FX + 16, TOP + 226, "One Euro Filter（以 ID 為 key）", 13.5, INK, "bold")
fps, T = 30, 2.4
ts = np.arange(0, T, 1 / fps)
truth = np.where(ts < 1.0, 0.0, 1.0) + 0.08 * np.sin(ts * 3)
noisy = truth + np.random.default_rng(7).normal(0, 0.06, ts.size)
flt = OneEuroFilter(0.8, 0.6)
smooth = np.array([flt(v, t) for v, t in zip(noisy, ts)])
ox, oy, ow, oh = FX + 18, TOP + 244, FW - 36, 96


def curve(ys, col, sw):
    lo, hi = -0.3, 1.3
    d = "M" + " L".join(f"{ox + t / T * ow:.1f},{oy + oh - (y - lo) / (hi - lo) * oh:.1f}" for t, y in zip(ts, ys))
    add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round"/>')


line(ox, oy + oh, ox + ow, oy + oh, GREY)
line(ox, oy, ox, oy + oh, GREY)
curve(noisy, MUTE, 1.1)
curve(smooth, fuse_col, 2.2)
line(ox + 60, oy + oh + 20, ox + 78, oy + oh + 20, MUTE, 1.5)
text(ox + 83, oy + oh + 24, "原始", 11, GREY)
line(ox + 122, oy + oh + 20, ox + 140, oy + oh + 20, fuse_col, 2.2)
text(ox + 145, oy + oh + 24, "平滑", 11, GREY)
lines(FX + 16, TOP + 392, [
    f"{it('f')}{sub('c')} = {it('f')}{sub('c,min')} + β·|{it('ẋ')}|",
    "慢動作強平滑、快動作低延遲",
    f"3D：{it('f')}{sub('c,min')} 0.8 Hz · β 0.6",
    "消失 0.5 s 後刪除該 ID 的狀態",
], 12, INK, gap=1.55)
text(FX + 16, TOP + 530, "同一 ID 同一顏色", 12, trk_col, "bold")

# ================================================================ g 輸出
GX, GW = 1515, 245
panel(GX, TOP - 18, "g", "輸出")
rect(GX, TOP, GW, 552, C["out"][1], r=10)
tw, th, gap = 104, 64, 9
for i, name in enumerate(["疊圖", "並排 3D", "深度", "僅骨架", "僅輪廓"]):
    x = GX + 14 + (i % 2) * (tw + gap)
    y = TOP + 16 + (i // 2) * (th + 26)
    rect(x, y, tw, th, {"疊圖": "#c9d3dc", "深度": "url(#g-depth)"}.get(name, "#121a20"), r=3)
    if name == "並排 3D":
        rect(x, y, tw / 2, th, "#c9d3dc", r=3)
        for k in range(4):
            line(x + tw / 2 + 4 + k * 12, y + th - 6, x + tw / 2 + 10 + k * 10, y + th - 24, "#3a4a52", 0.8)
        silhouette(body(x + 26, y + 36, 44), ID1, 5, 0.6)
        skeleton(body(x + 78, y + 32, 34), ID1, 1.1, dots=False)
    else:
        PA, PB = body(x + 36, y + 36, 46), body(x + 72, y + 38, 42, mirror=True)
        if name == "疊圖":
            silhouette(PA, ID1, 6, 0.6)
            silhouette(PB, ID2, 6, 0.6)
            skeleton(PA, "#163040", 1, dots=False)
            skeleton(PB, "#3a1020", 1, dots=False)
        elif name == "深度":
            silhouette(PA, "#c6e94a", 6)
            silhouette(PB, "#f6d34a", 6)
        elif name == "僅骨架":
            skeleton(PA, ID1, 1.3, dots=False)
            skeleton(PB, ID2, 1.3, dots=False)
        else:
            outline(PA, ID1, 7, "#121a20")
            outline(PB, ID2, 7, "#121a20")
    text(x + tw / 2, y + th + 15, name, 12, INK, anchor="middle")
x, y = GX + 14 + tw + gap, TOP + 16 + 2 * (th + 26)
rect(x, y, tw, th, "#ffffff", RULE, r=3)
lines(x + 8, y + 18, ["rgb.mp4", "depth/*.png", "skeleton.jsonl"], 11, INK, gap=1.45)
text(x + tw / 2, y + th + 15, "● 錄製", 12, INK, anchor="middle")
lines(GX + 14, TOP + 300, [
    "可拖曳旋轉的 3D 視圖",
    "（pyqtgraph OpenGL）",
    "每人卡片：ID、距離、",
    "實測關節比例",
    "錄製可用 --play 回放，",
    "skeleton.jsonl 可轉 CSV",
], 12, INK, gap=1.55)

# ================================================================ 主連線
split_x = AX + 262
arrow([(AX + 215, TOP + 233), (split_x, TOP + 233), (split_x, BY + 80), (BX - 4, BY + 80)], rgb_col)
line(split_x, TOP + 233, split_x, CY + 80, rgb_col, 2.2)
arrow([(split_x, CY + 80), (BX - 4, CY + 80)], rgb_col)
arrow([(AX + 215, TOP + 240), (AX + 248, TOP + 240), (AX + 248, DY + 66), (BX - 4, DY + 66)], depth_col)
for y0, col, label in ((BY + 80, rgb_col, "遮罩 + ID"), (CY + 80, rgb_col, "2D 關節"), (DY + 66, depth_col, "對齊深度")):
    arrow([(BX + BW, y0), (EX - 4, y0)], col)
    text((BX + BW + EX) / 2, y0 - 7, label, 11.5, col, anchor="middle")
arrow([(EX + EW, TOP + 260), (FX - 4, TOP + 260)], fuse_col)
text((EX + EW + FX) / 2, TOP + 252, "3D", 11.5, fuse_col, anchor="middle")
arrow([(FX + FW, TOP + 260), (GX - 4, TOP + 260)], out_col)

# ================================================================ h 每幀時序
HY = TOP + 630
line(AX, HY - 30, W - 40, HY - 30, RULE)
panel(AX, HY + 12, "h", "每幀時序（單人實測中位數 · Apple M5 · 1280×720）")
t = TIMINGS
x0, scale, row_h = AX + 280, 24, 34
y0 = HY + 50
budget = 1000 / 30
for i, (label, start, dur, col) in enumerate([
    ("執行緒 A · YOLO 分割 + 追蹤（GPU）", 0, t["segmentation"], rgb_col),
    ("執行緒 B · RTMPose 逐人（CoreML）", 0, t["pose"], rgb_col),
    ("主執行緒 · 深度對齊", 0, t["d2c"], depth_col),
    ("主執行緒 · 融合 + 3D + 平滑", t["segmentation"], t["fusion"], fuse_col),
]):
    y = y0 + i * row_h
    text(x0 - 12, y + 15, label, 12.5, INK, anchor="end")
    rect(x0 + start * scale, y + 2, max(dur * scale, 3), 18, col, r=3, extra='opacity="0.85"')
    text(x0 + (start + dur) * scale + 6, y + 16, f"{dur:.1f} ms", 11.5, GREY)
yb = y0 + 4 * row_h + 6
line(x0, yb, x0 + 35 * scale, yb, GREY)
for ms in range(0, 36, 5):
    line(x0 + ms * scale, yb, x0 + ms * scale, yb + 5, GREY)
    text(x0 + ms * scale, yb + 18, str(ms), 11, GREY, anchor="middle")
text(x0 + 35 * scale + 8, yb + 4, "ms", 11, GREY)
line(x0 + budget * scale, y0 - 10, x0 + budget * scale, yb, "#c0392b", 1.2, "4 3")
text(x0 + budget * scale + 4, y0 - 14, "30 fps 預算 33.3 ms", 11.5, "#c0392b")
line(x0 + t["total"] * scale, y0 - 10, x0 + t["total"] * scale, yb, INK, 1.2, "2 2")
text(x0 + t["total"] * scale - 4, y0 - 14, f"每幀 {t['total']:.1f} ms", 11.5, INK, "bold", "end")
text(x0, yb + 50, f"RTMPose 用上一幀的人框，才能與 YOLO 平行；三人時每幀 {t['total_3p']:.1f} ms（約 {1000 / t['total_3p']:.0f} fps）。", 12, GREY)

cx_, cy_ = 1330, HY + 74
for dx, dy, lab in ((56, 0, "x"), (0, 56, "y")):
    arrow([(cx_, cy_), (cx_ + dx, cy_ + dy)], INK, 1.5)
    text(cx_ + dx + (8 if dx else -4), cy_ + dy + (5 if dx else 18), lab, 13, INK, italic=True)
add(f'<circle cx="{cx_}" cy="{cy_}" r="7" fill="none" stroke="{INK}" stroke-width="1.5"/>')
add(f'<path d="M{cx_ - 5},{cy_ - 5} L{cx_ + 5},{cy_ + 5} M{cx_ - 5},{cy_ + 5} L{cx_ + 5},{cy_ - 5}" stroke="{INK}" stroke-width="1.3"/>')
text(cx_ - 13, cy_ - 11, "z", 13, INK, italic=True, anchor="end")
lines(cx_ + 90, cy_ - 2, ["RGB 相機座標系", "x 右、y 下、z 朝前（⊗ 指入場景）", "單位公尺"], 12.5, INK, gap=1.5)
lgx, lgy = 1250, HY + 186
for i, (col, lab) in enumerate(((rgb_col, "RGB / 2D 結果"), (depth_col, "深度"), (fuse_col, "3D 結果"), (out_col, "輸出"))):
    x = lgx + (i % 2) * 230
    y = lgy + (i // 2) * 28
    arrow([(x, y), (x + 36, y)], col, 2)
    text(x + 46, y + 4, lab, 12.5, INK)

text(AX, H - 20, "示意圖。人形取自骨架模型對範例影像的實際輸出（第二人為鏡像複製）；One Euro 曲線為合成訊號經本專案濾波器的實際計算；"
     "時序為本機實測；ID 顏色與程式中一致。", 11, MUTE)

add("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote {OUT}")
