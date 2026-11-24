"""Thanh trạng thái dưới cùng: thời gian xử lý từng bước, thông báo, số lượt đã ghi."""
from __future__ import annotations

import customtkinter as ctk

from app.ui import texts, theme


class StatusBar(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self._left = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w")
        self._left.pack(side="left")
        self._right = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED, anchor="e")
        self._right.pack(side="right")
        self._message_until = 0.0

    def set_timings(self, timings: dict) -> None:
        if timings and not self._message_active():
            text = (f"Tìm tờ {timings.get('detect_ms', 0):.0f} ms   ·   "
                    f"Nắn phẳng {timings.get('rectify_ms', 0):.0f} ms   ·   "
                    f"Phân loại {timings.get('classify_ms', 0):.0f} ms")
            if timings.get("hidden"):
                text += "   ·   " + texts.STATUS_HIDDEN.format(n=timings["hidden"])
            self._left.configure(text=text)

    def set_session(self, n_entries: int, grand_total: str) -> None:
        self._right.configure(text=f"Đã ghi {n_entries} lượt · {grand_total}" if n_entries else "")

    def flash(self, text: str, color: str = theme.TEXT, seconds: float = 4.0) -> None:
        """Hiện một thông báo ngắn (ghi nhận xong, lỗi...) trong vài giây."""
        import time
        self._message_until = time.time() + seconds
        self._left.configure(text=text, text_color=color)

    def _message_active(self) -> bool:
        import time
        if time.time() < self._message_until:
            return True
        self._left.configure(text_color=theme.MUTED)
        return False
