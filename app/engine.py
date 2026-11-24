"""Engine: cầu nối giữa lõi xử lý (app/core) và giao diện (app/ui).

Engine chạy camera -> Pipeline -> Aggregator ở MỘT LUỒNG NỀN, nên cửa sổ không bị đơ.
Giao diện chỉ cần:

    engine = Engine()
    engine.start_camera(0)            # hoặc start_camera("video.mp4"), start_image(ảnh BGR)
    out = engine.latest()             # Output mới nhất (gọi bao nhiêu lần cũng được)
    out.frame, out.state              # khung hình BGR và FrameState (tổng tiền, danh sách tờ)
    engine.pop_events()               # các sự kiện "vừa ổn định" để đọc to một lần
    engine.set_manual(track_id, "50000")   # sửa tay một tờ (None = trả về tự động)
    engine.reset()                    # làm lại lượt
    engine.stop()

Muốn đổi cách tìm tờ / phân loại thì sửa trong app/core, không cần đụng file này.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.core.aggregator import Aggregator, FrameState
from app.core.camera import Camera
from app import config
from app.core import model_registry
from app.core.classifier import CurrencyClassifier
from app.core.pipeline import Pipeline


@dataclass(frozen=True)
class Output:
    """Một "ảnh chụp trạng thái" mà luồng nền đưa cho giao diện."""
    frame: np.ndarray | None          # khung BGR gốc (chưa vẽ gì lên)
    state: FrameState                 # kết quả tổng hợp: tổng tiền, các tờ, cảnh báo...
    timings: dict = field(default_factory=dict)   # detect_ms / rectify_ms / classify_ms
    fps: float = 0.0
    frame_id: int = 0
    ended: bool = False               # video đã phát hết
    message: str = ""                 # lỗi/ghi chú từ luồng nền (nếu có)


@dataclass(frozen=True)
class StableEvent:
    """Phát ra đúng một lần khi tổng tiền vừa ổn định (dùng để đọc to)."""
    total: int
    n_notes: int


class Engine:
    def __init__(self) -> None:
        self.error: str | None = None             # lỗi nạp mô hình (None = sẵn sàng)
        self.classifier: CurrencyClassifier | None = None
        self.pipeline: Pipeline | None = None
        self.aggregator = Aggregator()
        self.events: "queue.SimpleQueue[StableEvent]" = queue.SimpleQueue()

        self._lock = threading.Lock()             # bảo vệ aggregator (luồng nền + giao diện cùng dùng)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera: Camera | None = None
        self._out = Output(None, self.aggregator.state)

        self.models: list[model_registry.ModelInfo] = model_registry.discover()
        try:
            self.classifier = CurrencyClassifier(self._default_model_path())
            self.classifier.warmup()
            self.pipeline = Pipeline(self.classifier)
        except Exception as exc:  # noqa: BLE001 - hiển thị nguyên văn lỗi cho người dùng
            self.error = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ trạng thái
    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> Output:
        return self._out

    def pop_events(self) -> list[StableEvent]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------------ điều khiển nguồn
    def start_camera(self, source=0, loop_video: bool = False) -> None:
        """Mở webcam (số) hoặc file video (đường dẫn). Ném RuntimeError nếu không mở được."""
        self._require_ready()
        self.stop()
        cam = Camera(source=source, loop_video=loop_video).start()   # có thể ném RuntimeError
        self._camera = cam
        self._begin(cam=cam, image=None)

    def start_image(self, image_bgr: np.ndarray) -> None:
        """Phân tích một ảnh tĩnh (lặp lại vài khung để bỏ phiếu ổn định)."""
        self._require_ready()
        self.stop()
        self._begin(cam=None, image=image_bgr)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._camera is not None:
            self._camera.stop()
            self._camera = None

    # ------------------------------------------------------------------ thao tác trên kết quả
    def set_manual(self, track_id: int, label: str | None) -> None:
        with self._lock:
            self.aggregator.set_manual(track_id, label)
            prev = self._out          # cập nhật ngay để giao diện thấy kết quả kể cả khi đang dừng
            self._out = Output(prev.frame, self.aggregator.state, prev.timings, prev.fps, prev.frame_id)

    def reset(self) -> None:
        with self._lock:
            self.aggregator.reset()
            self._out = Output(self._out.frame, self.aggregator.state, self._out.timings,
                               self._out.fps, self._out.frame_id)

    # ------------------------------------------------------------------ chọn mô hình
    def _default_model_path(self):
        """Mô hình mặc định trong config nếu có, không thì mô hình đầu tiên tìm được."""
        if config.CLASSIFIER_PATH.exists() or not self.models:
            return config.CLASSIFIER_PATH
        return self.models[0].path

    @property
    def current_model(self) -> model_registry.ModelInfo | None:
        if self.classifier is None:
            return None
        return model_registry.describe(self.classifier.model_path)

    def load_model(self, path) -> None:
        """Đổi sang mô hình khác khi đang chạy. Ném lỗi (giữ nguyên mô hình cũ) nếu nạp không được.

        Giữ ngưỡng tin cậy và TTA đang chọn; làm lại lượt đếm vì kết quả của hai mô hình không trộn được.
        """
        old = self.classifier
        new = CurrencyClassifier(
            path,
            conf_threshold=old.conf_threshold if old else None,
            use_tta=old.use_tta if old else None,
        )
        new.warmup()
        with self._lock:
            self.classifier = new
            if self.pipeline is None:
                self.pipeline = Pipeline(new)
            else:
                self.pipeline.classifier = new
            self.error = None
            self.pipeline.reset()
            self.aggregator.reset()
            self._out = Output(self._out.frame, self.aggregator.state, self._out.timings,
                               self._out.fps, self._out.frame_id)

    # ------------------------------------------------------------------ tham số chỉnh lúc đang chạy
    def set_conf_threshold(self, value: float) -> None:
        if self.classifier is not None:
            self.classifier.conf_threshold = float(value)

    def set_hide_not_money(self, enabled: bool) -> None:
        """Bật/tắt việc ẩn vật mà bộ phân loại xác nhận không phải tiền."""
        if self.pipeline is not None:
            self.pipeline.hide_not_money = bool(enabled)

    def set_tta(self, enabled: bool) -> None:
        if self.classifier is not None:
            self.classifier.use_tta = bool(enabled)

    # ------------------------------------------------------------------ nội bộ
    def _require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(self.error or "Chưa nạp được mô hình")

    def _begin(self, cam: Camera | None, image: np.ndarray | None) -> None:
        with self._lock:
            self.aggregator.reset()
            if self.pipeline is not None:
                self.pipeline.reset()                 # quên các tờ đang theo dõi + kết luận cũ của nguồn trước
            self._out = Output(self._out.frame, self.aggregator.state)   # xoá cờ "đã hết video" của lượt trước
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cam, image), daemon=True, name="engine")
        self._thread.start()

    def _run(self, cam: Camera | None, image: np.ndarray | None) -> None:
        frame_id = 0
        try:
            while not self._stop.is_set():
                if image is not None:
                    frame = image.copy()
                    time.sleep(0.05)                 # ~20 khung/giây cho ảnh tĩnh
                    fps = 20.0
                else:
                    frame, _ = cam.read(timeout=0.5)
                    if frame is None:
                        if cam.ended:
                            self._publish(frame_id, None, ended=True)
                            return
                        continue
                    fps = cam.fps
                frame_id += 1
                detections, timings = self.pipeline.process(frame)
                with self._lock:
                    state = self.aggregator.update(detections)
                if state.just_stabilized:
                    self.events.put(StableEvent(state.total, state.n_money))
                self._out = Output(frame, state, timings, fps, frame_id)
        except Exception as exc:  # noqa: BLE001
            self._publish(frame_id, f"{type(exc).__name__}: {exc}", ended=True)

    def _publish(self, frame_id: int, message: str | None, ended: bool) -> None:
        prev = self._out
        self._out = Output(prev.frame, prev.state, prev.timings, prev.fps, frame_id,
                           ended=ended, message=message or "")
