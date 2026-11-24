"""Thẻ lớn hiển thị TỔNG TIỀN, số tờ và dòng gợi ý trạng thái."""
from __future__ import annotations

import customtkinter as ctk

from app.core.aggregator import FrameState
from app.ui import texts, theme


class TotalCard(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=theme.ACCENT, corner_radius=theme.RADIUS)
        self._title = ctk.CTkLabel(self, text=texts.TOTAL_TITLE, font=theme.font(12, True),
                                   text_color="#DDE6FF", anchor="w")
        self._title.pack(anchor="w", padx=20, pady=(14, 0))
        self._total = ctk.CTkLabel(self, text="0 ₫", font=theme.font(40, True),
                                   text_color=theme.ON_ACCENT, anchor="w")
        self._total.pack(anchor="w", padx=20)
        self._count = ctk.CTkLabel(self, text="", font=theme.font(13), text_color="#DDE6FF", anchor="w")
        self._count.pack(anchor="w", padx=20)
        self._hint = ctk.CTkLabel(self, text=texts.HINT_EMPTY, font=theme.font(12),
                                  text_color="#DDE6FF", anchor="w", justify="left", wraplength=340)
        self._hint.pack(anchor="w", padx=20, pady=(2, 14))

    def update_state(self, state: FrameState | None) -> None:
        if state is None or state.n_tracks == 0:
            self._set(0, "", texts.HINT_EMPTY, theme.ACCENT)
            return
        parts = [f"{state.n_money} tờ được tính"]
        if state.n_unrecognized:
            parts.append(f"{state.n_unrecognized} chưa chắc")
        if state.n_warning:
            parts.append(f"{state.n_warning} cảnh báo")
        if state.has_warning:
            hint, color = texts.HINT_WARNING, theme.DANGER
        elif state.stable:
            hint, color = texts.HINT_STABLE, theme.GOOD
        elif state.n_not_money == state.n_tracks:
            hint, color = texts.HINT_NOT_MONEY, theme.ACCENT
        else:
            hint, color = texts.HINT_WORKING, theme.ACCENT
        self._set(state.total, "  ·  ".join(parts), hint, color)

    def _set(self, total: int, count: str, hint: str, color: str) -> None:
        main, sub = ("#06261C", "#0B3D2E") if color == theme.GOOD else (theme.ON_ACCENT, "#DDE6FF")
        self.configure(fg_color=color)
        self._title.configure(text_color=sub)
        self._total.configure(text_color=main)
        self._count.configure(text_color=sub)
        self._hint.configure(text_color=sub)
        self._total.configure(text=texts.fmt_vnd(total))
        self._count.configure(text=count)
        self._hint.configure(text=hint)
