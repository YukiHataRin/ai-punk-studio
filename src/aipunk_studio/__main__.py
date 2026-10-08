"""python -m aipunk_studio [--headless] [--camera NAME] [--ws] [--start] [--play DIR] [--export DIR] [--list-cameras]"""

import argparse
import sys

from . import __version__, utf8_stdio
from .config import load_config


def main():
    utf8_stdio()
    parser = argparse.ArgumentParser(description="AI Punk Studio：RGB-D 人體分割、多人 3D 骨架與 ID 追蹤")
    parser.add_argument("--version", action="version", version=f"aipunk-studio {__version__}")
    parser.add_argument("--config", help="設定檔路徑（預設 config/default.toml）")
    parser.add_argument("--start", action="store_true", help="開啟後立即開始擷取")
    parser.add_argument("--no-seg", action="store_true", help="關閉 YOLO 人體分割，只跑骨架")
    parser.add_argument("--cv", action="store_true", help="改用 OpenCV 檢視器（開發用）")
    parser.add_argument("--list-cameras", action="store_true", help="列出相機名稱與裝置 ID")
    parser.add_argument("--play", metavar="DIR", help="回放錄製目錄（recordings/...），不需要相機")
    parser.add_argument("--export", metavar="DIR", help="把錄製目錄的骨架與指標轉成 CSV 後結束")
    parser.add_argument("--headless", action="store_true", help="不開視窗，只擷取、計算並以 WebSocket 推送（Ctrl-C 結束）")
    parser.add_argument("--ws", action="store_true", help="介面模式也開啟 WebSocket 串流")
    parser.add_argument("--no-ws", action="store_true", help="headless 模式不開 WebSocket（例如只錄製）")
    parser.add_argument("--ws-host", help="WebSocket 綁定位址（預設 127.0.0.1；0.0.0.0 開放區域網路）")
    parser.add_argument("--ws-port", type=int, help="WebSocket 埠號（預設 8765）")
    parser.add_argument("--record", action="store_true", help="headless 模式啟動即錄製（含 RGB-D 影像）")
    parser.add_argument("--duration", type=float, help="headless 模式執行幾秒後自動結束")
    parser.add_argument("--camera", metavar="NAME", help="預選攝影機（名稱、裝置 ID 或 --list-cameras 的編號）；非 Astra 時 3D 為估計")
    parser.add_argument("--screenshot", metavar="PATH", help=argparse.SUPPRESS)  # 開發用：延遲後存視窗截圖並結束
    parser.add_argument("--screenshot-delay", type=float, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--view", choices=["overlay", "split", "depth", "skeleton", "contour"], help="初始畫面模式")
    parser.add_argument("--tab", choices=["overview", "metrics"], help="初始分頁：總覽或指標")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.ws_host:
        cfg["stream"]["host"] = args.ws_host
    if args.ws_port is not None:  # 0 = 自動選可用埠，不能用真假判斷
        cfg["stream"]["port"] = args.ws_port
    if args.ws:
        cfg["stream"]["enabled"] = True

    if args.export:
        from .io.export import export_session
        try:
            results = export_session(args.export)
        except FileNotFoundError as e:
            sys.exit(str(e))
        for path, rows in results:
            print(f"已輸出 {rows} 列：{path}")
        return
    if args.list_cameras:
        from .sensors.discovery import discover_cameras
        for camera in discover_cameras():
            print(f"{camera.index}\t{camera.name}\t{camera.uid}")
        return
    if args.headless:
        from .apps.headless import run as run_headless
        sys.exit(run_headless(cfg, segmentation=not args.no_seg, camera=args.camera, play=args.play,
                              stream=not args.no_ws, record=args.record, duration=args.duration))
    if args.cv:
        from .apps.cv_viewer import run
        run(cfg, segmentation=not args.no_seg)
        return

    from .apps.gui import run as run_gui
    sys.exit(run_gui(cfg, segmentation=not args.no_seg, play=args.play, camera=args.camera, view=args.view,
                     tab=args.tab, start=args.start, screenshot=args.screenshot,
                     screenshot_delay=args.screenshot_delay))


if __name__ == "__main__":
    main()
