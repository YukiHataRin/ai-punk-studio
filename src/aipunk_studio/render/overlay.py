"""2D 疊圖：人體遮罩、骨架、ID 標籤。全部在 RGB 原始解析度上繪製。

合併自 Human Mask Studio 的 compose_masks 與本專案的 draw_2d。
"""

import cv2
import numpy as np

from .colors import track_bgr

MIN_VIS = 0.3


def draw_masks(img, seg_people, opacity=0.55, mask_only=False):
    """回傳新影像。mask_only=True 時為黑底只有彩色遮罩。"""
    out = np.zeros_like(img) if mask_only else img.copy()
    for person in seg_people:
        color = np.array(track_bgr(person.track_id), np.uint8)
        m = person.mask
        if mask_only:
            out[m] = color
        else:
            out[m] = (out[m] * (1 - opacity) + color * opacity).astype(np.uint8)
    return out


def draw_contours(img, seg_people, thickness=2):
    """只畫每個人遮罩的外輪廓（就地修改 img）。"""
    for person in seg_people:
        contours, _ = cv2.findContours(person.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, track_bgr(person.track_id), thickness, cv2.LINE_AA)


def draw_label(img, text, x, y, color):
    """在 (x, y) 上方畫有底色的標籤，會自動夾在畫面內。"""
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    h = size[1] + baseline + 10
    left = max(0, min(int(x), img.shape[1] - size[0] - 14))
    top = max(0, int(y) - h)
    bottom = min(img.shape[0] - 1, top + h)
    cv2.rectangle(img, (left, top), (left + size[0] + 14, bottom), color, -1)
    cv2.putText(img, text, (left + 7, bottom - baseline - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 13, 15), 2, cv2.LINE_AA)


def draw_skeleton(img, pose, skeleton, color):
    """實心點 = 深度實測，空心點 = 推估。頭部不畫（fmt.hidden）。"""
    px = pose.pixels.astype(int)
    vis = pose.visibility
    fmt = pose.fmt
    for a, b in fmt.shown_connections:
        if min(vis[a], vis[b]) > MIN_VIS:
            cv2.line(img, tuple(px[a]), tuple(px[b]), color, 2, cv2.LINE_AA)
    for j in fmt.shown:
        if vis[j] > MIN_VIS:
            filled = skeleton is not None and skeleton.measured[j]
            cv2.circle(img, tuple(px[j]), 5, color, -1 if filled else 2, cv2.LINE_AA)


def draw_people(img, people, skeleton=True, labels=True):
    """people：[TrackedPerson]。畫骨架，並在頭頂（沒有骨架時在框上方）標示 ID 與距離。"""
    for person in people:
        color = track_bgr(person.track_id)
        pose = person.pose
        if pose is not None:
            if skeleton:
                draw_skeleton(img, pose, person.skeleton, color)
            h = pose.fmt.head
            anchor = pose.pixels[h] if pose.visibility[h] > MIN_VIS else pose.pixels[list(pose.fmt.shoulders)].mean(0)
            x, y = anchor[0] - 30, anchor[1] - 24
        elif person.segment is not None:
            x, y = person.segment.box[:2]
        else:
            continue
        if not labels:
            continue
        dist = person.distance_text()
        # OpenCV 字型沒有「≈」，改用「~」
        draw_label(img, f"ID {person.track_id}" + (f"  {dist.replace('≈', '~')}" if dist else ""), x, y, color)


def draw_depth(img, reg_depth, max_mm=4000, alpha=0.6):
    """把對齊後的深度圖（任意解析度）以 TURBO 色彩疊到 img 上，用來檢查對齊。"""
    d = cv2.resize(reg_depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = cv2.applyColorMap(np.clip(d / max_mm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blend = cv2.addWeighted(img, 1 - alpha, vis, alpha, 0)
    return np.where(d[..., None] > 0, blend, img)
