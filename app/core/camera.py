"""Đọc camera / file video ở thread nền (luôn giữ khung mới nhất) + bộ đếm FPS.

    cam = Camera(source=0).start()
    frame, frame_id = cam.read(timeout=1.0)   # chờ khung mới hơn lần đọc trước
    cam.fps                                    # FPS thực tế của luồng đọc
    cam.stop()
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from app import config


def parse_source(source) -> int | str:
    """'0' -> 0 (webcam số 0); chuỗi khác (đường dẫn video, URL) giữ nguyên."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    return int(s) if s.isdigit() else s


class Camera:
    def __init__(self, source=config.CAMERA_SOURCE, width: int = config.CAMERA_WIDTH,
                 height: int = config.CAMERA_HEIGHT, fps: int = config.CAMERA_FPS,
                 lock_exposure: bool = config.LOCK_EXPOSURE, loop_video: bool = False,
                 focus: float | None = config.FOCUS_VALUE, exposure: float | None = config.EXPOSURE_VALUE,
                 show_dialog: bool = False):
        self.source = parse_source(source)
        self.width, self.height, self.target_fps = width, height, fps
        self.lock_exposure = lock_exposure
        self.loop_video = loop_video
        self.focus = focus                # khác None: tắt autofocus, khoá nét ở giá trị này
        self.exposure = exposure          # khác None: tắt tự phơi sáng, đặt thời gian phơi sáng
        self.show_dialog = show_dialog    # Windows: mở hộp thoại thuộc tính của driver camera
        self.warnings: list[str] = []     # thuộc tính mà driver từ chối (cap.set trả về False)
        self.is_file = isinstance(self.source, str) and os.path.exists(self.source)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._last_read_id = 0
        self._times: deque[float] = deque(maxlen=30)
        self.ended = False
        self.error: str | None = None

    # ------------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        if os.name == "nt" and isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # mở nhanh và ổn định hơn trên Windows
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được camera/video: {self.source!r}")
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # cần để webcam đạt HD 30 FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.lock_exposure:  # tuỳ hệ điều hành / driver, thất bại thì bỏ qua
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.warnings = []
            if self.focus is not None:  # autofocus dò nét khi tay cử động gây nhoè -> khoá nét ở khoảng cách cầm tiền
                if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
                    self.warnings.append("driver không cho tắt autofocus")
                if not cap.set(cv2.CAP_PROP_FOCUS, float(self.focus)):
                    self.warnings.append("driver không cho đặt giá trị focus")
            if self.exposure is not None:  # phơi sáng ngắn: bớt nhoè do rung tay, giữ được FPS khi thiếu sáng
                if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):
                    self.warnings.append("driver không cho tắt tự phơi sáng")
                if not cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure)):
                    self.warnings.append("driver không cho đặt thời gian phơi sáng")
            if self.show_dialog and os.name == "nt" and isinstance(self.source, int):
                cap.set(cv2.CAP_PROP_SETTINGS, 1)  # chỉnh focus/phơi sáng bằng tay rồi ghi lại số vào config.py
        return cap

    def start(self) -> "Camera":
        if self._thread and self._thread.is_alive():
            return self
        self._cap = self._open()
        self._stop.clear()
        self.ended = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-reader")
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        file_fps = self._cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0
        delay = 1.0 / file_fps if file_fps and file_fps > 1 else 1.0 / 30
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                if self.is_file and self.loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.ended = True
                with self._cond:
                    self._cond.notify_all()
                break
            now = time.perf_counter()
            self._times.append(now)
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()
            if self.is_file:  # phát video theo tốc độ gốc
                time.sleep(max(0.0, delay - (time.perf_counter() - t0)))

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray | None, int]:
        """Chờ khung mới hơn lần đọc trước. Trả về (None, id) nếu hết thời gian hoặc video đã hết."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._frame_id == self._last_read_id and not self.ended:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None, self._last_read_id
                self._cond.wait(remaining)
            if self._frame_id == self._last_read_id:
                return None, self._last_read_id
            self._last_read_id = self._frame_id
            return self._frame, self._frame_id

    @property
    def fps(self) -> float:
        times = list(self._times)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None