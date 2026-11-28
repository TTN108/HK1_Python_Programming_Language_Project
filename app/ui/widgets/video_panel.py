"""Khung hiển thị video. Nhận ảnh BGR (đã vẽ kết quả) và tự co theo kích thước khung."""
from __future__ import annotations

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image
from typing import Callable

from app.ui import texts, theme


class VideoPanel(ctk.CTkFrame):
    def __init__(self, parent, on_click: Callable[[int, int], None] | None = None,   # đổi
                 on_undo: Callable[[], None] | None = None):
        super().__init__(parent, fg_color=theme.VIDEO_BG, corner_radius=theme.RADIUS_SM)
        # place() không làm khung "phình" theo ảnh -> tránh vòng lặp ảnh lớn dần
        self._label = ctk.CTkLabel(self, text=texts.VIDEO_PLACEHOLDER, font=theme.font(16),
                                   text_color=theme.MUTED, fg_color="transparent")
        self._label.place(relx=0.5, rely=0.5, anchor="center")
        self._image = None          # giữ tham chiếu để ảnh không bị thu hồi bộ nhớ
        self._on_click, self._on_undo = on_click, on_undo     # mới
        self._geom: tuple[int, int, int, int] | None = None   # mới: (w, h, rộng, cao) khi hiển thị
        self._pickable = False                                # mới
        for widget in (self, self._label):                    # mới
            widget.bind("<Button-1>", self._clicked)
            widget.bind("<Button-3>", self._right_clicked)


    def show(self, frame_bgr: np.ndarray) -> None:
        avail_w = max(self.winfo_width() - 8, 160)
        avail_h = max(self.winfo_height() - 8, 120)
        h, w = frame_bgr.shape[:2]
        scale = min(avail_w / w, avail_h / h)
        size = (max(int(w * scale), 1), max(int(h * scale), 1))
        self._geom = (w, h, size[0], size[1])                 # mới
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._image = ctk.CTkImage(light_image=Image.fromarray(rgb), size=size)
        self._label.configure(image=self._image, text="")

    def show_message(self, text: str) -> None:
        """Quay về màn hình chữ (chưa có video, hoặc báo lỗi)."""
        self._image = None
        self._label.configure(image=None, text=text)

    def set_pickable(self, on: bool) -> None:
        self._pickable = on

    def _clicked(self, event) -> None:
        if not self._pickable or self._on_click is None or self._geom is None:
            return
        w, h, dw, dh = self._geom
        px = event.x_root - self.winfo_rootx() - (self.winfo_width() - dw) / 2    # ảnh nằm giữa khung
        py = event.y_root - self.winfo_rooty() - (self.winfo_height() - dh) / 2
        x, y = px * w / dw, py * h / dh
        if 0 <= x < w and 0 <= y < h:
            self._on_click(int(x), int(y))

    def _right_clicked(self, event) -> None:
        if self._pickable and self._on_undo is not None:
            self._on_undo()