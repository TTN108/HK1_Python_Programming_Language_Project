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
import json                                                      # mới
from collections import deque                                    # mới
from dataclasses import dataclass, field, replace                # đổi: thêm replace
import cv2

import numpy as np

from app.core.aggregator import Aggregator, Donation, FrameState
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
        self.donations: "queue.SimpleQueue[Donation]" = queue.SimpleQueue()   # mới: lượt mới cho giao diện (âm thanh cảm ơn)
        self.recent: deque = deque(maxlen=config.DASHBOARD_RECENT)             # mới
        self.audit_error: str | None = None                                    # mới
        self._session = time.strftime("%Y%m%d_%H%M%S")                         # mới
        
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

    # ------------------------------------------------------------------ trạm quyên góp (mới)
    @property
    def slot_ready(self) -> bool:
        return self.aggregator.slot is not None
    @property
    def slot(self):                                  # mới: vùng khe (None nếu chưa chọn) để giao diện vẽ
        return self.aggregator.slot
    
    def set_slot(self, points) -> None:
        """Đặt vùng khe (pixel của khung gốc). Chỉ một lần, khoá tới khi chạy lại chương trình."""
        with self._lock:
            self.aggregator.set_slot(points)

    def pop_donations(self) -> list[Donation]:
        out = []
        while True:
            try:
                out.append(self.donations.get_nowait())
            except queue.Empty:
                return out

    def dashboard(self) -> dict:
        """Dữ liệu cho Live Dashboard công khai (chỉ có số, không có ảnh)."""
        with self._lock:
            st = self.aggregator.state
            recent = [dict(seq=d.seq, value=d.value, review=d.review,
                           time=time.strftime("%H:%M:%S", time.localtime(d.t))) for d in self.recent]
        return {"total": st.donated_total, "count": st.donation_count, "recent": recent,
                "slot_ready": self.slot_ready, "running": self.running, "session": self._session}

    def _save_snapshot(self, d: Donation) -> str:
        """Lưu ảnh chụp lượt đóng góp (vẽ khung tờ + số tiền); trả tên file, '' nếu lỗi."""
        if d.snapshot is None:
            return ""
        try:
            img = d.snapshot.copy()
            cv2.polylines(img, [np.round(d.corners).astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
            slot = self.aggregator.slot
            if config.SNAPSHOT_CROP and slot is not None:
                pts = np.vstack([d.corners, slot]).astype(np.int32)
                x1, y1 = np.maximum(pts.min(axis=0) - config.SNAPSHOT_PAD_PX, 0)
                x2, y2 = pts.max(axis=0) + config.SNAPSHOT_PAD_PX
                img = img[y1:y2, x1:x2]
            cv2.putText(img, f"#{d.seq} {d.value} VND  total {d.total_after}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_QUALITY])
            if not ok:
                return ""
            config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            name = f"{self._session}_{d.seq:05d}_{d.value}.jpg"
            buf.tofile(str(config.SNAPSHOT_DIR / name))       # ghi được cả đường dẫn có dấu
            return name
        except Exception as exc:  # noqa: BLE001
            self.audit_error = f"ảnh chụp: {type(exc).__name__}: {exc}"
            return ""

    def _log_donation(self, d: Donation) -> Donation:
        """Mỗi lượt: một ảnh chụp + một dòng JSON Lines (chỉ ghi thêm). Trả bản nhẹ, không giữ ảnh."""
        self.audit_error = None
        name = self._save_snapshot(d)
        rec = dict(session=self._session, seq=d.seq,
                   time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.t)),
                   label=d.label, value=d.value, confidence=round(d.confidence, 3), manual=d.manual,
                   total_after=d.total_after, review=d.review, snapshot=name)
        try:
            config.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(config.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.audit_error = f"nhật ký: {type(exc).__name__}: {exc}"
        return replace(d, snapshot=None, snapshot_file=name)
    
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
                    state = self.aggregator.update(detections, frame)          # đổi: truyền thêm frame
                for d in state.new_donations:                                  # mới
                    light = self._log_donation(d)
                    with self._lock:
                        self.recent.append(light)
                    self.donations.put(light)
                if state.just_stabilized and not config.KIOSK_MODE:            # đổi: kiosk không đọc to tổng "đang trong khung"
                    self.events.put(StableEvent(state.total, state.n_money))
                self._out = Output(frame, state, timings, fps, frame_id)
        except Exception as exc:  # noqa: BLE001
            self._publish(frame_id, f"{type(exc).__name__}: {exc}", ended=True)

    def _publish(self, frame_id: int, message: str | None, ended: bool) -> None:
        prev = self._out
        self._out = Output(prev.frame, prev.state, prev.timings, prev.fps, frame_id,
                           ended=ended, message=message or "")
