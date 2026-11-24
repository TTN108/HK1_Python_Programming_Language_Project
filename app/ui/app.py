"""Cửa sổ chính: ghép các widget lại và nối với Engine / Speaker / SessionLog.

Muốn THÊM/BỚT/ĐỔI VỊ TRÍ một khối giao diện -> sửa hàm _build_layout() ở đây.
Muốn đổi NỘI DUNG một khối -> sửa file tương ứng trong app/ui/widgets/.
Logic xử lý ảnh không nằm ở đây (xem app/core/ và app/engine.py).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np

from app import config
from app.core import model_registry
from app.core.aggregator import COUNTED
from app.engine import Engine
from app.services.speech import Speaker
from app.services.storage import SessionLog, save_snapshot
from app.ui import texts, theme
from app.ui.overlay import draw_tracks
from app.ui.widgets.action_bar import ActionBar
from app.ui.widgets.control_bar import ControlBar
from app.ui.widgets.header import Header
from app.ui.widgets.note_list import NoteList
from app.ui.widgets.settings_panel import SettingsPanel
from app.ui.widgets.status_bar import StatusBar
from app.ui.widgets.total_card import TotalCard
from app.ui.widgets.video_panel import VideoPanel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TICK_MS = 33          # ~30 lần/giây cập nhật giao diện


class App(ctk.CTk):
    def __init__(self, engine: Engine, speaker: Speaker, session: SessionLog,
                 auto_source: str | None = None):
        theme.apply_global_theme()
        super().__init__()
        self.engine, self.speaker, self.session = engine, speaker, session
        self._last_state = None
        self._last_id = -1
        self._running_ui = False

        self.title(texts.WINDOW_TITLE)
        self.geometry(theme.WINDOW_SIZE)
        self.minsize(*theme.WINDOW_MIN)
        self.configure(fg_color=theme.BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._init_state()
        self.after(TICK_MS, self._tick)
        if auto_source is not None:
            self.after(300, lambda: self._start_source(auto_source))

    # ------------------------------------------------------------------ bố cục
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = Header(self)
        self.header.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 8))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=24)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # --- cột trái: video + điều khiển nguồn ---------------------------------
        left = ctk.CTkFrame(body, fg_color=theme.PANEL, corner_radius=theme.RADIUS)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self.video = VideoPanel(left)
        self.video.grid(row=0, column=0, sticky="nsew", padx=14, pady=(14, 10))
        self.controls = ControlBar(left, on_toggle=self._toggle, on_open=self._open_file,
                                   on_snapshot=self._snapshot)
        self.controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))

        # --- cột phải: kết quả ----------------------------------------------------
        right = ctk.CTkFrame(body, fg_color=theme.PANEL, corner_radius=theme.RADIUS, width=theme.SIDEBAR_WIDTH)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        self.total_card = TotalCard(right)
        self.total_card.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        self.note_list = NoteList(right, on_manual=self.engine.set_manual)
        self.note_list.grid(row=1, column=0, sticky="nsew", padx=10)
        conf = self.engine.classifier.conf_threshold if self.engine.classifier else config.CONF_THRESHOLD
        self.settings = SettingsPanel(right, conf=conf, sound=self.speaker.enabled, tta=config.USE_TTA,
                                      on_conf=self.engine.set_conf_threshold,
                                      on_sound=self._set_sound, on_tta=self.engine.set_tta,
                                      model_names=[m.name for m in self.engine.models],
                                      model_current=self.engine.classifier.name if self.engine.classifier else None,
                                      on_model=self._switch_model,
                                      hide=config.HIDE_NOT_MONEY, on_hide=self.engine.set_hide_not_money)
        self.settings.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 8))
        self.actions = ActionBar(right, on_commit=self._commit, on_reset=self._reset, on_export=self._export)
        self.actions.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))

        self.status = StatusBar(self)
        self.status.grid(row=2, column=0, sticky="ew", padx=28, pady=(6, 10))

    def _init_state(self) -> None:
        if self.engine.ready:
            self._show_model(self.engine.current_model)
        else:
            self.header.set_model_status(False)
            self.video.show_message(
                "⚠ Chưa nạp được mô hình nhận diện\n\n"
                f"{self.engine.error}\n\n"
                f"Kiểm tra file {config.CLASSIFIER_PATH.name} trong thư mục models/ và gói onnxruntime.")

    # ------------------------------------------------------------------ mô hình
    def _show_model(self, info: model_registry.ModelInfo | None) -> None:
        """Cập nhật huy hiệu ở tiêu đề và dòng chú thích trong cài đặt theo mô hình đang dùng."""
        if info is None:
            return
        self.header.set_model_status(True, info.name)
        self.settings.set_model(info.name)
        if info.trusted:
            self.settings.set_model_note(texts.MODEL_INFO.format(
                n=len(info.classes), src=model_registry.SOURCE_TEXT[info.classes_source], mb=info.size_mb))
        else:
            self.settings.set_model_note(texts.MODEL_UNTRUSTED.format(name=info.name), warn=True)

    def _switch_model(self, name: str) -> None:
        info = next((m for m in self.engine.models if m.name == name), None)
        if info is None:
            return
        try:
            self.engine.load_model(info.path)
        except Exception as exc:  # noqa: BLE001 - lỗi nạp mô hình hiển thị nguyên văn, giữ mô hình cũ
            self.status.flash(texts.MODEL_SWITCH_FAILED.format(err=f"{type(exc).__name__}: {exc}"), theme.DANGER, 8.0)
            self._show_model(self.engine.current_model)       # trả ô chọn về mô hình đang dùng
            return
        self._show_model(info)
        self.status.flash(texts.MODEL_SWITCHED.format(name=name), theme.GOOD)

    # ------------------------------------------------------------------ vòng lặp cập nhật giao diện
    def _tick(self) -> None:
        out = self.engine.latest()
        if out.state is not self._last_state or out.frame_id != self._last_id:
            self._last_state, self._last_id = out.state, out.frame_id
            if out.frame is not None:
                self.video.show(draw_tracks(out.frame, out.state))
            self.total_card.update_state(out.state)
            self.note_list.update_state(out.state)
            self.header.set_fps(out.fps)
            self.status.set_timings(out.timings)

        for event in self.engine.pop_events():          # tổng vừa ổn định -> đọc to đúng một lần
            self.speaker.say_total(event.total, event.n_notes)

        if out.ended and not self.engine.running and self._running_ui:
            self._finish_source(out.message)
        self.after(TICK_MS, self._tick)

    # ------------------------------------------------------------------ nguồn ảnh
    def _toggle(self) -> None:
        if self._running_ui:
            self._stop_source()
        else:
            self._start_source(self.controls.camera_index())

    def _start_source(self, source) -> None:
        try:
            self.engine.start_camera(source)
        except RuntimeError as exc:
            self.video.show_message(f"⚠ {exc}")
            self.status.flash(str(exc), theme.DANGER)
            return
        self._set_running(True)

    def _stop_source(self) -> None:
        self.engine.stop()
        self._set_running(False)

    def _finish_source(self, message: str) -> None:
        """Video hết hoặc luồng nền gặp lỗi."""
        self.engine.stop()
        self._set_running(False)
        self.status.flash(message or "Đã hết video", theme.WARN if not message else theme.DANGER, 6.0)

    def _set_running(self, running: bool) -> None:
        self._running_ui = running
        self.controls.set_running(running)

    def _open_file(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Chọn ảnh hoặc video",
            filetypes=[("Ảnh / video", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                       ("Tất cả", "*.*")])
        if not path:
            return
        if Path(path).suffix.lower() in IMAGE_EXTS:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)   # đọc được đường dẫn có dấu
            if img is None:
                self.status.flash("Không đọc được ảnh này", theme.DANGER)
                return
            try:
                self.engine.start_image(img)
            except RuntimeError as exc:
                self.status.flash(str(exc), theme.DANGER)
                return
            self._set_running(True)
        else:
            self._start_source(path)

    def _snapshot(self) -> None:
        out = self.engine.latest()
        if out.frame is None:
            self.status.flash("Chưa có khung hình để chụp", theme.WARN)
            return
        try:
            path = save_snapshot(draw_tracks(out.frame, out.state))
        except Exception as exc:  # noqa: BLE001
            self.status.flash(f"Không lưu được ảnh: {exc}", theme.DANGER)
            return
        self.status.flash(f"Đã lưu ảnh: {path}", theme.GOOD)

    # ------------------------------------------------------------------ thao tác trên kết quả
    def _commit(self) -> None:
        state = self.engine.latest().state
        counts = Counter(v.value for v in state.tracks if v.status in COUNTED)
        if not counts:
            self.status.flash("Chưa có tờ nào được xác nhận để ghi nhận", theme.WARN)
            return
        entry = self.session.add(state.total, dict(counts))
        self._refresh_session()
        self.status.flash(f"Đã ghi nhận lượt #{len(self.session.entries)}: "
                          f"{texts.fmt_vnd(entry.total)} ({entry.n_notes} tờ)", theme.GOOD)

    def _reset(self) -> None:
        self.engine.reset()
        self.status.flash("Đã làm lại lượt đếm", theme.MUTED)

    def _export(self) -> None:
        if not self.session.entries:
            self.status.flash("Chưa có lượt nào để xuất — hãy bấm Ghi nhận lượt trước", theme.WARN)
            return
        from tkinter import filedialog
        config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Xuất lịch sử", initialdir=str(config.EXPORT_DIR), defaultextension=".csv",
            initialfile="kiem_dem.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.status.flash(f"Đã xuất: {self.session.export_csv(path)}", theme.GOOD)

    def _refresh_session(self) -> None:
        self.status.set_session(len(self.session.entries), texts.fmt_vnd(self.session.grand_total))

    def _set_sound(self, enabled: bool) -> None:
        self.speaker.enabled = enabled

    # ------------------------------------------------------------------ đóng cửa sổ
    def _on_close(self) -> None:
        self.engine.stop()
        self.speaker.close()
        self.destroy()


def run(source: str | None = None, sound: bool = True) -> None:
    engine = Engine()
    speaker = Speaker(enabled=sound and config.SOUND_ENABLED)
    session = SessionLog()
    App(engine, speaker, session, auto_source=source).mainloop()
