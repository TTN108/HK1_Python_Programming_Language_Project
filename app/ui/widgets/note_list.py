"""Danh sách từng tờ + ô chọn sửa tay mệnh giá.

Mỗi dòng ứng với một "track" của Aggregator. Dòng được TẠO MỘT LẦN rồi cập nhật tại chỗ (không dựng lại
mỗi khung hình) nên không nhấp nháy và ô chọn không bị đóng giữa chừng.

    note_list = NoteList(parent, on_manual=callback)     # callback(track_id, label | None)
    note_list.update_state(state)
"""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from app import config
from app.core.aggregator import COUNTED, FrameState, TrackView
from app.ui import texts, theme


def _option_values() -> tuple[list[str], dict[str, str | None]]:
    """Danh sách hiển thị trong ô chọn + bảng tra ngược: chữ hiển thị -> tên lớp (None = tự động)."""
    classes = [c for c in config.CLASSES if c != config.NOT_MONEY_CLASS]
    classes.sort(key=config.class_to_value, reverse=True)
    lookup: dict[str, str | None] = {texts.AUTO_LABEL: None}
    for c in classes:
        lookup[texts.fmt_vnd(config.class_to_value(c))] = c
    lookup[texts.NOT_MONEY_LABEL] = config.NOT_MONEY_CLASS
    return list(lookup.keys()), lookup


class _Row(ctk.CTkFrame):
    def __init__(self, parent, track_id: int, values: list[str], on_change: Callable[[str], None]):
        super().__init__(parent, fg_color=theme.CARD, corner_radius=theme.RADIUS_SM)
        self.track_id = track_id
        self._cache: tuple | None = None
        self.grid_columnconfigure(2, weight=1)

        self._bar = ctk.CTkFrame(self, width=6, height=44, corner_radius=3, fg_color=theme.MUTED)
        self._bar.grid(row=0, column=0, rowspan=2, padx=(10, 10), pady=8, sticky="ns")
        ctk.CTkLabel(self, text=f"#{track_id}", font=theme.font(12), text_color=theme.MUTED,
                     width=28).grid(row=0, column=1, rowspan=2)
        self._title = ctk.CTkLabel(self, text="", font=theme.font(16, True), text_color=theme.TEXT, anchor="w")
        self._title.grid(row=0, column=2, sticky="sw", padx=(4, 0), pady=(8, 0))
        self._sub = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w")
        self._sub.grid(row=1, column=2, sticky="nw", padx=(4, 0), pady=(0, 8))
        self._menu = ctk.CTkOptionMenu(
            self, values=values, width=124, height=30, font=theme.font(12), dropdown_font=theme.font(12),
            fg_color=theme.PANEL, button_color=theme.PANEL, button_hover_color=theme.CARD_HOVER,
            dropdown_fg_color=theme.CARD, dropdown_hover_color=theme.CARD_HOVER,
            dropdown_text_color=theme.TEXT, text_color=theme.TEXT, command=on_change)
        self._menu.grid(row=0, column=3, rowspan=2, padx=10)

    def refresh(self, v: TrackView, menu_text: str) -> None:
        key = (v.status, v.label, v.value, v.guess, round(v.confidence, 2), v.visible, v.reasons, menu_text, v.donated)
        if key == self._cache:
            return
        self._cache = key
        color = theme.STATUS_COLOR.get(v.status, theme.MUTED)
        if v.status in COUNTED:
            title = texts.fmt_vnd(v.value)
            color = theme.DENOM_COLOR.get(v.value, color)
        elif v.status == "not_money":
            title = texts.NOT_MONEY_LABEL
        elif v.guess and v.guess != config.NOT_MONEY_CLASS:
            title = f"{texts.fmt_vnd(config.class_to_value(v.guess))} ?"
        else:
            title = "…"
        sub = texts.STATUS_TEXT.get(v.status, v.status)
        warn = [texts.REASON_TEXT.get(r, r) for r in v.reasons if r in config.WARNING_REASONS]
        if warn:
            sub += " (" + ", ".join(warn) + ")"
        if v.donated:                                                     # mới
            sub = texts.DONATED_TEXT
        elif not v.visible:                                               # đổi: if -> elif
            sub += texts.LOST_SUFFIX
        self._bar.configure(fg_color=color)
        self._title.configure(text=title, text_color=theme.TEXT if v.visible else theme.MUTED)
        self._sub.configure(text=sub)
        self._menu.set(menu_text)
        self._menu.configure(state="disabled" if v.donated else "normal")


class NoteList(ctk.CTkFrame):
    def __init__(self, parent, on_manual: Callable[[int, str | None], None]):
        super().__init__(parent, fg_color="transparent")
        self._on_manual = on_manual
        self._values, self._lookup = _option_values()
        self._display_of = {label: text for text, label in self._lookup.items() if label is not None}
        self._rows: dict[int, _Row] = {}
        self._order: tuple[int, ...] = ()

        ctk.CTkLabel(self, text=texts.LIST_TITLE, font=theme.font(14, True), text_color=theme.TEXT,
                     anchor="w").pack(anchor="w", padx=6, pady=(0, 4))
        self._scroll = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        self._scroll.pack(fill="both", expand=True)
        self._empty = ctk.CTkLabel(self._scroll, text=texts.LIST_EMPTY, font=theme.font(13),
                                   text_color=theme.MUTED)

    def update_state(self, state: FrameState | None) -> None:
        tracks = state.tracks if state else []
        ids = {v.id for v in tracks}
        for tid in [t for t in self._rows if t not in ids]:        # xoá dòng của tờ đã biến mất
            self._rows.pop(tid).destroy()
        if not tracks:
            self._order = ()
            self._empty.pack(pady=28)
            return
        self._empty.pack_forget()
        for v in tracks:                                            # tracks đã sắp trái -> phải
            row = self._rows.get(v.id)
            if row is None:
                row = _Row(self._scroll, v.id, self._values,
                           on_change=lambda choice, tid=v.id: self._on_manual(tid, self._lookup[choice]))
                self._rows[v.id] = row
            row.refresh(v, self._menu_text(v))
        order = tuple(v.id for v in tracks)
        if order != self._order:                                    # chỉ xếp lại khi thứ tự đổi (đỡ nhấp nháy)
            self._order = order
            for tid in order:
                self._rows[tid].pack_forget()
                self._rows[tid].pack(fill="x", pady=3, padx=2)

    def _menu_text(self, v: TrackView) -> str:
        if not v.manual:
            return texts.AUTO_LABEL
        if v.label == config.NOT_MONEY_CLASS:
            return texts.NOT_MONEY_LABEL
        return self._display_of.get(v.label, texts.AUTO_LABEL)
