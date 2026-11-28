"""Thanh thao tác trên kết quả (cuối cột phải): Ghi nhận lượt, Làm lại, Xuất CSV."""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from app.ui import texts, theme
from app.ui.widgets.control_bar import make_button


class ActionBar(ctk.CTkFrame):
    def __init__(self, parent, on_commit: Callable, on_reset: Callable, on_export: Callable,
                 on_slot: Callable | None = None, on_dashboard: Callable | None = None,   # mới
                 kiosk: bool = False):                                                     # mới
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure((0, 1), weight=1, uniform="a")
        self._btn_slot = None                                                              # mới
        if kiosk:                                                                          # mới
            self._btn_slot = make_button(self, texts.BTN_SLOT, on_slot, "primary")
            self._btn_slot.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
            make_button(self, texts.BTN_DASHBOARD, on_dashboard, "good").grid(
                row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))
        else:                                                                              # đổi: dòng cũ nằm trong else
            make_button(self, texts.BTN_COMMIT, on_commit, "good").grid(row=0, column=0, columnspan=2,
                                                                          sticky="ew", pady=(0, 8))
        # hai nút Làm lại / Xuất CSV giữ nguyên
        make_button(self, texts.BTN_RESET, on_reset).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        make_button(self, texts.BTN_EXPORT, on_export).grid(row=1, column=1, sticky="ew", padx=(4, 0))

    def set_slot_locked(self, locked: bool) -> None:                                       # mới
        if self._btn_slot is not None:
            self._btn_slot.configure(text=texts.BTN_SLOT_LOCKED if locked else texts.BTN_SLOT,
                                     state="disabled" if locked else "normal")