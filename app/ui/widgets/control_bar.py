"""Thanh điều khiển nguồn ảnh (dưới khung video): Bắt đầu/Dừng, Mở tệp, Chụp ảnh, chọn camera."""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from app.ui import texts, theme


def make_button(parent, text: str, command: Callable, kind: str = "neutral", width: int = 0):
    """Nút dùng chung cho toàn giao diện. kind: neutral | primary | good | danger."""
    colors = {
        "neutral": (theme.CARD, theme.CARD_HOVER, theme.TEXT),
        "primary": (theme.ACCENT, theme.ACCENT_HOVER, theme.ON_ACCENT),
        "good": (theme.GOOD, theme.GOOD_HOVER, "#06261C"),
        "danger": (theme.DANGER, theme.DANGER_HOVER, theme.ON_ACCENT),
    }[kind]
    kwargs = {"width": width} if width else {}
    return ctk.CTkButton(parent, text=text, command=command, fg_color=colors[0], hover_color=colors[1],
                         text_color=colors[2], corner_radius=theme.RADIUS_SM, height=40,
                         font=theme.font(13, True), **kwargs)


class ControlBar(ctk.CTkFrame):
    def __init__(self, parent, on_toggle: Callable, on_open: Callable, on_snapshot: Callable):
        super().__init__(parent, fg_color="transparent")
        self._btn_toggle = make_button(self, texts.BTN_START, on_toggle, "primary", width=128)
        self._btn_toggle.pack(side="left", padx=(0, 8))
        make_button(self, texts.BTN_OPEN, on_open).pack(side="left", padx=4)
        make_button(self, texts.BTN_SNAPSHOT, on_snapshot).pack(side="left", padx=4)

        self._cam = ctk.CTkOptionMenu(
            self, values=["0", "1", "2", "3"], width=64, height=34, font=theme.font(12),
            fg_color=theme.CARD, button_color=theme.CARD, button_hover_color=theme.CARD_HOVER,
            dropdown_fg_color=theme.CARD, dropdown_hover_color=theme.CARD_HOVER,
            dropdown_text_color=theme.TEXT, text_color=theme.TEXT)
        self._cam.pack(side="right")
        ctk.CTkLabel(self, text="Camera", font=theme.font(12), text_color=theme.MUTED).pack(side="right", padx=8)

    def set_running(self, running: bool) -> None:
        if running:
            self._btn_toggle.configure(text=texts.BTN_STOP, fg_color=theme.DANGER, hover_color=theme.DANGER_HOVER,
                                       text_color=theme.ON_ACCENT)
        else:
            self._btn_toggle.configure(text=texts.BTN_START, fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                                       text_color=theme.ON_ACCENT)

    def camera_index(self) -> int:
        return int(self._cam.get())
