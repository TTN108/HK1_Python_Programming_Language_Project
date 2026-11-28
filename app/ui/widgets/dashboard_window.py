"""Màn hình công khai (Live Dashboard): tổng quyên góp, số lượt, lời cảm ơn, các lượt gần nhất.

Chỉ hiển thị SỐ lấy từ engine.dashboard(), không có ảnh chụp. Kéo cửa sổ sang màn hình / máy chiếu thứ hai
rồi bấm F11 để toàn màn hình (Esc để thoát).

    win = DashboardWindow(parent)
    win.update_data(engine.dashboard())    # gọi vài lần/giây, chỉ vẽ lại khi dữ liệu đổi
    win.show_thanks(donation)              # biểu ngữ cảm ơn vài giây
"""
from __future__ import annotations

import time

import customtkinter as ctk

from app.ui import texts, theme


class DashboardWindow(ctk.CTkToplevel):
    THANKS_SECONDS = 4.0

    def __init__(self, parent):
        super().__init__(parent)
        self.title(texts.DASH_WINDOW_TITLE)
        self.geometry("1000x640")
        self.configure(fg_color=theme.BG)
        self._cache: tuple | None = None
        self._thanks_until = 0.0
        self.bind("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.after(200, self.lift)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(self, text=texts.DASH_TITLE, font=theme.font(30, True),
                     text_color=theme.TEXT).grid(row=0, column=0, pady=(28, 0))
        ctk.CTkLabel(self, text=texts.DASH_SUBTITLE, font=theme.font(14),
                     text_color=theme.MUTED).grid(row=1, column=0)
        self._total = ctk.CTkLabel(self, text=texts.fmt_vnd(0), font=theme.font(96, True), text_color=theme.GOOD)
        self._total.grid(row=2, column=0, pady=(18, 0))
        self._count = ctk.CTkLabel(self, text="", font=theme.font(20), text_color=theme.MUTED)
        self._count.grid(row=3, column=0, sticky="n")
        self._banner = ctk.CTkLabel(self, text="", font=theme.font(28, True), height=64,
                                    corner_radius=theme.RADIUS, fg_color="transparent")
        self._banner.grid(row=4, column=0, sticky="ew", padx=60, pady=(0, 12))
        self._recent = ctk.CTkLabel(self, text="", font=theme.font(16), text_color=theme.TEXT, justify="left")
        self._recent.grid(row=5, column=0, pady=(0, 24))

    def update_data(self, data: dict) -> None:
        key = (data["total"], data["count"], data["slot_ready"],
               tuple((r["seq"], r["value"], r["review"]) for r in data["recent"]))
        if key != self._cache:
            self._cache = key
            self._total.configure(text=texts.fmt_vnd(data["total"]))
            self._count.configure(text=texts.DASH_COUNT.format(n=data["count"]) if data["slot_ready"]
                                  else texts.DASH_NOT_READY)
            lines = []
            for r in reversed(data["recent"]):                 # mới nhất ở trên cùng
                amount = texts.DASH_PENDING if r["review"] else "+" + texts.fmt_vnd(r["value"])
                lines.append(f"#{r['seq']}    {r['time']}    {amount}")
            self._recent.configure(text="\n".join(lines))
        if self._thanks_until and time.time() >= self._thanks_until:
            self._thanks_until = 0.0
            self._banner.configure(text="", fg_color="transparent")

    def show_thanks(self, d) -> None:
        if d.value > 0:
            self._banner.configure(text=texts.DASH_THANKS.format(value=texts.fmt_vnd(d.value)),
                                   fg_color=theme.GOOD, text_color="#06261C")
        else:
            self._banner.configure(text=texts.DASH_THANKS_PENDING, fg_color=theme.ACCENT, text_color=theme.ON_ACCENT)
        self._thanks_until = time.time() + self.THANKS_SECONDS