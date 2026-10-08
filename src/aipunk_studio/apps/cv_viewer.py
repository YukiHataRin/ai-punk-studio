"""OpenCV 檢視器：左邊 RGB + 遮罩 + 2D 骨架，右邊 3D 骨架。Qt 介面完成前的開發用工具。

按鍵：
  q / Esc  離開
  m        切換人體遮罩
  t        切換 BoT-SORT 追蹤（關閉時改用骨架位置配對 ID）
  b        切換背景（RGB 影像 / 黑底）
  o        疊上對齊後的深度圖（檢查 RGB-深度對齊）
  [ / ]    外參 x 平移 -/+ 2 mm（數值印在終端機）
  r        重設 3D 視角
  s        存截圖到 captures/
"""

import time

import cv2
import numpy as np

from ..config import PROJECT_ROOT
from ..pipeline.pipeline import Pipeline
from ..render.overlay import draw_depth, draw_masks, draw_people
from ..render.view3d import Orbit3DView
from ..sensors.astra import AstraSource

WINDOW = "AI Punk Studio (OpenCV)"


def run(cfg, segmentation=True):
    from ..sensors.discovery import find_astra_rgb_index
    cfg["rgb"]["index"] = find_astra_rgb_index(cfg["rgb"]["index"])  # 插拔其他攝影機後編號會變
    source = AstraSource(cfg).start()
    pipeline = Pipeline(cfg, segmentation=segmentation)

    disp_w = cfg["view"]["rgb_display_width"]
    disp_h = int(cfg["rgb"]["height"] * disp_w / cfg["rgb"]["width"])
    view3d = Orbit3DView(cfg["view"]["panel_size"])
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, f, _: view3d.on_mouse(e, x - disp_w, y, f) if x >= disp_w else None)

    show_rgb, show_depth, show_mask = True, False, segmentation
    last_stamp, fps, n, t0 = 0.0, 0.0, 0, time.time()
    canvas = None
    try:
        while True:
            if source.error:
                raise RuntimeError(source.error)
            frame, stamp, depth = source.latest()
            if frame is not None and stamp != last_stamp:
                last_stamp = stamp
                out = pipeline.process(frame, depth, stamp)

                left = frame if show_rgb else np.zeros_like(frame)
                if show_mask:
                    left = draw_masks(left, out.segmentation.people, cfg["segmentation"]["opacity"], mask_only=not show_rgb)
                if show_depth:
                    left = draw_depth(left, out.reg_depth)
                left = left.copy() if left is frame else left  # 不在擷取執行緒共用的影格上直接畫
                draw_people(left, out.people)
                left = cv2.resize(left, (disp_w, disp_h))

                n += 1
                if time.time() - t0 >= 1.0:
                    fps, n, t0 = n / (time.time() - t0), 0, time.time()
                t = out.timings
                cv2.putText(left, f"{fps:.1f} fps  people: {len(out.people)}  seg {t['segmentation']:.0f} ms  pose {t['pose']:.0f} ms",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                right = view3d.render(out.people)
                h = max(disp_h, right.shape[0])
                canvas = np.zeros((h, disp_w + right.shape[1], 3), np.uint8)
                canvas[:disp_h, :disp_w] = left
                canvas[:right.shape[0], disp_w:] = right
                cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m") and segmentation:
                show_mask = not show_mask
            elif key == ord("t") and segmentation:
                cfg["segmentation"]["tracking"] = not cfg["segmentation"]["tracking"]
                pipeline.reset_tracking()
                print(f"BoT-SORT 追蹤：{'開' if pipeline.tracking else '關'}")
            elif key == ord("b"):
                show_rgb = not show_rgb
            elif key == ord("o"):
                show_depth = not show_depth
            elif key in (ord("["), ord("]")):
                t = pipeline.registration.t
                t[0] += 0.002 if key == ord("]") else -0.002
                print(f"extrinsics.translation = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            elif key == ord("r"):
                view3d.reset()
            elif key == ord("s") and canvas is not None:
                path = PROJECT_ROOT / "captures" / f"studio_{time.strftime('%Y%m%d_%H%M%S')}.png"
                path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(path), canvas)
                print(f"已存檔：{path}")
    finally:
        pipeline.close()
        source.stop()
        cv2.destroyAllWindows()
