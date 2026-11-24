"""Thanh thao tác trên kết quả (cuối cột phải): Ghi nhận lượt, Làm lại, Xuất CSV."""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from app.ui import texts, theme
from app.ui.widgets.control_bar import make_button


class ActionBar(ctk.CTkFrame):
    def __init__(self, parent, on_commit: Callable, on_reset: Callable, on_export: Callable):
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure((0, 1), weight=1, uniform="a")
        make_button(self, texts.BTN_COMMIT, on_commit, "good").grid(row=0, column=0, columnspan=2,
                                                                      sticky="ew", pady=(0, 8))
        make_button(self, texts.BTN_RESET, on_reset).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        make_button(self, texts.BTN_EXPORT, on_export).grid(row=1, column=1, sticky="ew", padx=(4, 0))
