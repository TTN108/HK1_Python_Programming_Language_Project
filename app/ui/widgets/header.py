"""Thanh tiêu đề: tên ứng dụng, trạng thái mô hình, FPS."""
from __future__ import annotations

import customtkinter as ctk

from app.ui import texts, theme


class Header(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        titles = ctk.CTkFrame(self, fg_color="transparent")
        titles.pack(side="left")
        ctk.CTkLabel(titles, text=texts.APP_TITLE, font=theme.font(26, True),
                     text_color=theme.TEXT, anchor="w").pack(anchor="w")
        ctk.CTkLabel(titles, text=texts.APP_SUBTITLE, font=theme.font(12),
                     text_color=theme.MUTED, anchor="w").pack(anchor="w")

        self._badge = ctk.CTkLabel(self, text="", font=theme.font(12), fg_color=theme.CARD,
                                   corner_radius=14, padx=12, pady=4)
        self._badge.pack(side="left", padx=18)

        self._fps = ctk.CTkLabel(self, text="", font=theme.font(12), text_color=theme.MUTED)
        self._fps.pack(side="right")

    def set_model_status(self, ok: bool, detail: str = "") -> None:
        text = f"●  {texts.MODEL_OK if ok else texts.MODEL_ERROR}"
        self._badge.configure(text=text, text_color=theme.GOOD if ok else theme.DANGER)
        if detail:
            self._badge.configure(text=f"{text}: {detail[:70]}")

    def set_fps(self, fps: float) -> None:
        self._fps.configure(text=f"{fps:0.0f} FPS" if fps else "")
