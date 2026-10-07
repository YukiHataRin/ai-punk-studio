"""背景擷取執行緒與「只留最新一張」的影格信箱。"""

import sys
import threading
import time

import cv2

# OpenCV 擷取後端：macOS AVFoundation、Windows Media Foundation、Linux V4L2
CAPTURE_BACKEND = {"darwin": cv2.CAP_AVFOUNDATION, "win32": cv2.CAP_MSMF}.get(sys.platform, cv2.CAP_V4L2)


class LatestFrame:
    """單格信箱：擷取端只覆寫最新影格，不會累積過時的畫面。"""

    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.stamp = 0.0
        self.closed = False

    def put(self, frame):
        with self.condition:
            self.frame, self.stamp = frame, time.monotonic()
            self.condition.notify_all()

    def peek(self):
        """取最新影格但不清空（多個消費者共用時使用）。"""
        with self.condition:
            return self.frame, self.stamp

    def take(self, timeout=0.1):
        """等待並取走新影格；逾時或已關閉時回傳 None。"""
        with self.condition:
            self.condition.wait_for(lambda: self.frame is not None or self.closed, timeout)
            frame, self.frame = self.frame, None
            return frame

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class Grabber:
    """在背景執行緒反覆呼叫 read_once()，把結果放進 LatestFrame。"""

    name = "grabber"

    def __init__(self):
        self.frames = LatestFrame()
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=self.name)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.frames.close()
        self._thread.join(timeout=2)
        self.close()

    def latest(self):
        return self.frames.peek()

    def read_once(self):
        """回傳一張影格；暫時沒有時回傳 None。"""
        raise NotImplementedError

    def close(self):
        """釋放裝置。"""

    def _loop(self):
        try:
            while not self._stop.is_set():
                frame = self.read_once()
                if frame is None:
                    self._stop.wait(0.005)
                    continue
                self.frames.put(frame)
        except Exception as error:  # 交給上層顯示
            self.error = str(error)
            self.frames.close()
