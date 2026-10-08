"""Headless 伺服器：不開視窗，只擷取、計算並以 WebSocket 推送（可同時錄製）。Ctrl-C 結束。

    python -m aipunk_studio --headless                       # 自動選 Astra Pro，ws://127.0.0.1:8765
    python -m aipunk_studio --headless --camera "j5 WebCam JVCU100" --ws-host 0.0.0.0
    python -m aipunk_studio --headless --play recordings/20261006_112336 --duration 10
"""

import signal
import sys
import threading
import time
from pathlib import Path

from ..io.stream import StreamServer
from ..pipeline.runner import CaptureRunner
from ..sensors.discovery import apply_camera, discover_cameras, select_camera

STATS_INTERVAL = 5.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cfg, segmentation=True, camera=None, play=None, stream=True, record=False, duration=None):
    """回傳結束代碼（0 = 正常）。"""
    source_factory = None
    if play:
        from ..sensors.playback import PlaybackSource
        source_factory = lambda c, path=Path(play): PlaybackSource(path)  # noqa: E731
        log(f"來源：回放 {play}")
    else:
        cameras = discover_cameras()
        cam = select_camera(cameras, camera)
        if cam is None:
            names = "、".join(f"{c.index}:{c.name}" for c in cameras) or "（沒有）"
            log(f"找不到攝影機「{camera}」。可用的攝影機：{names}" if camera else "沒有偵測到攝影機")
            return 2
        apply_camera(cfg, cam)
        log(f"來源：{cam.name}（{'RGB + 深度' if cam.is_astra else '僅 RGB，3D 為估計'}）")

    server = None
    if stream:
        s = cfg["stream"]
        server = StreamServer(s["host"], s["port"], s).start()
        log(f"WebSocket 串流：{server.url}" + ("（區域網路可連線）" if s["host"] not in ("127.0.0.1", "localhost") else ""))

    stats = {"frames": 0, "people": 0, "fps": 0.0, "last": None}

    def on_output(out, metrics):
        stats["frames"] += 1
        stats["people"] = len(out.people)
        stats["fps"] = metrics["fps"]
        now = time.monotonic()
        if stats["last"] is None:  # 從第一幀開始計時，避免一開始印出 0 fps
            stats["last"] = now
        elif now - stats["last"] >= STATS_INTERVAL:
            stats["last"] = now
            clients = server.client_count if server else 0
            rec = f" · 錄製 {metrics['recorded_frames']} 幀" if metrics.get("recorded_frames") is not None else ""
            log(f"{stats['fps']:.1f} fps · {stats['people']} 人 · {clients} 個用戶端{rec}")

    runner = CaptureRunner(cfg, segmentation, source_factory,
                           on_status=log, on_ready=log, on_output=on_output,
                           on_recording=lambda path, on: log(f"{'開始錄製' if on else '錄製已儲存'}：{path}"),
                           publisher=server)
    if record:
        runner.update(record={"with_images": True})

    def request_stop(*_):
        if not runner.stopping:
            log("停止中…")
        runner.stop()

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    timer = None
    if duration:
        timer = threading.Timer(duration, request_stop)
        timer.daemon = True
        timer.start()
    code = 0
    try:
        runner.run()
    except Exception as error:
        log(f"錯誤：{error}")
        code = 1
    finally:
        if timer:
            timer.cancel()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if server:
            server.stop()
        log(f"結束，共處理 {stats['frames']} 幀")
    return code


if __name__ == "__main__":
    from ..config import load_config
    sys.exit(run(load_config()))
