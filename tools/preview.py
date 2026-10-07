"""Orbbec Astra Pro 即時預覽：左邊 RGB、右邊深度。

RGB 走 UVC（OpenCV / AVFoundation），深度走 pyorbbecsdk（OpenNI 協定）。
macOS 上 RGB 需要相機權限，請在 Claude 的終端機分頁或「終端機」App 執行：

    python preview.py

按鍵：q / Esc 離開、s 存一張 RGB + 深度快照、c 切換色彩映射。
滑鼠移到深度畫面上會顯示該點距離（mm）。
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import Config, Context, OBFormat, OBLogLevel, OBSensorType, Pipeline

WIDTH, HEIGHT = 640, 480
COLORMAPS = [cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_BONE]

mouse_xy = None


def on_mouse(event, x, y, flags, param):
    global mouse_xy
    # 只在右半邊（深度畫面）記錄座標
    mouse_xy = (x - WIDTH, y) if WIDTH <= x < WIDTH * 2 and 0 <= y < HEIGHT else None


def open_depth():
    Context.set_logger_level(OBLogLevel.NONE)
    pipe = Pipeline()
    profiles = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    cfg = Config()
    cfg.enable_stream(profiles.get_video_stream_profile(WIDTH, HEIGHT, OBFormat.Y11, 30))
    pipe.start(cfg)
    return pipe


def open_rgb(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(
            f"無法開啟 RGB 相機 {index}，請確認相機權限，並在終端機分頁執行。"
        )
    return cap


def read_depth_mm(pipe):
    frames = pipe.wait_for_frames(100)
    depth = frames.get_depth_frame() if frames else None
    if depth is None:
        return None
    data = np.frombuffer(depth.get_data(), dtype=np.uint16)
    data = data.reshape(depth.get_height(), depth.get_width())
    return (data * depth.get_depth_scale()).astype(np.uint16)


def colorize(depth_mm, max_mm, cmap):
    norm = np.clip(depth_mm.astype(np.float32) / max_mm, 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    vis[depth_mm == 0] = 0  # 無效深度顯示為黑色
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rgb-index", type=int, default=None, help="RGB 相機編號（預設自動尋找 Astra Pro HD Camera）")
    parser.add_argument("--max-mm", type=int, default=4000, help="深度色彩映射的最大距離（mm）")
    parser.add_argument("--out", type=Path, default=Path("captures"), help="快照存放資料夾")
    args = parser.parse_args()

    pipe = open_depth()
    if args.rgb_index is None:
        from astra_studio.pipeline.worker import find_astra_rgb_index
        args.rgb_index = find_astra_rgb_index(0)
    cap = open_rgb(args.rgb_index)

    win = "Astra Pro  |  RGB  |  Depth"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    cmap_i = 0
    depth_mm = np.zeros((HEIGHT, WIDTH), np.uint16)
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    t0, n, fps = time.time(), 0, 0.0

    try:
        while True:
            ok, frame = cap.read()
            if ok:
                rgb = cv2.resize(frame, (WIDTH, HEIGHT)) if frame.shape[:2] != (HEIGHT, WIDTH) else frame
            d = read_depth_mm(pipe)
            if d is not None:
                depth_mm = d if d.shape == (HEIGHT, WIDTH) else cv2.resize(d, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

            depth_vis = colorize(depth_mm, args.max_mm, COLORMAPS[cmap_i])
            if mouse_xy is not None:
                x, y = mouse_xy
                cv2.drawMarker(depth_vis, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
                cv2.putText(depth_vis, f"{depth_mm[y, x]} mm", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            n += 1
            if time.time() - t0 >= 1.0:
                fps, n, t0 = n / (time.time() - t0), 0, time.time()
            view = np.hstack([rgb, depth_vis])
            cv2.putText(view, f"{fps:.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                cmap_i = (cmap_i + 1) % len(COLORMAPS)
            if key == ord("s"):
                args.out.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(args.out / f"{stamp}_rgb.png"), rgb)
                cv2.imwrite(str(args.out / f"{stamp}_depth.png"), depth_mm)  # 16-bit，單位 mm
                cv2.imwrite(str(args.out / f"{stamp}_depth_vis.png"), depth_vis)
                print(f"已存檔：{args.out}/{stamp}_*.png")
    finally:
        cap.release()
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
