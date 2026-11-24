"""Cài đặt nhanh: chọn mô hình, ngưỡng tin cậy, đọc to, TTA. Các giá trị khác chỉnh trong app/config.py."""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from app.ui import texts, theme


class SettingsPanel(ctk.CTkFrame):
    def __init__(self, parent, conf: float, sound: bool, tta: bool,
                 on_conf: Callable[[float], None], on_sound: Callable[[bool], None],
                 on_tta: Callable[[bool], None],
                 model_names: list[str] | None = None, model_current: str | None = None,
                 on_model: Callable[[str], None] | None = None,
                 hide: bool = True, on_hide: Callable[[bool], None] | None = None):
        super().__init__(parent, fg_color=theme.CARD, corner_radius=theme.RADIUS_SM)
        self._on_conf, self._on_sound, self._on_tta = on_conf, on_sound, on_tta
        self._on_model, self._on_hide = on_model, on_hide

        ctk.CTkLabel(self, text=texts.SETTINGS_TITLE, font=theme.font(13, True), text_color=theme.TEXT,
                     anchor="w").grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 2))

        # --- chọn mô hình ---------------------------------------------------------
        ctk.CTkLabel(self, text=texts.SETTING_MODEL, font=theme.font(12), text_color=theme.MUTED,
                     anchor="w").grid(row=1, column=0, columnspan=2, sticky="w", padx=14)
        names = model_names or [texts.MODEL_NONE]
        self._model_menu = ctk.CTkOptionMenu(
            self, values=names, command=self._model_changed, font=theme.font(12), dynamic_resizing=False,
            fg_color=theme.PANEL, button_color=theme.ACCENT, text_color=theme.TEXT,
            state="normal" if model_names else "disabled")
        self._model_menu.set(model_current if model_current in names else names[0])
        self._model_menu.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 0))
        self._model_note = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED,
                                        anchor="w", justify="left", wraplength=320)
        self._model_note.grid(row=3, column=0, columnspan=2, sticky="w", padx=14, pady=(2, 8))

        # --- ngưỡng tin cậy ---------------------------------------------------------
        ctk.CTkLabel(self, text=texts.SETTING_CONF, font=theme.font(12), text_color=theme.MUTED,
                     anchor="w").grid(row=4, column=0, sticky="w", padx=14)
        self._conf_value = ctk.CTkLabel(self, text=f"{conf:.0%}", font=theme.font(12, True),
                                        text_color=theme.GOOD, width=44)
        self._conf_value.grid(row=4, column=1, sticky="e", padx=14)
        self._slider = ctk.CTkSlider(self, from_=0.4, to=0.95, number_of_steps=55, command=self._conf_changed,
                                     progress_color=theme.ACCENT, button_color=theme.GOOD,
                                     button_hover_color=theme.GOOD_HOVER, fg_color=theme.PANEL)
        self._slider.set(conf)
        self._slider.grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 6))

        self._sound = self._switch(texts.SETTING_SOUND, sound, 6, self._sound_changed)
        self._tta = self._switch(texts.SETTING_TTA, tta, 7, self._tta_changed)
        self._hide = self._switch(texts.SETTING_HIDE, hide, 8, self._hide_changed)
        self._hide.grid_configure(pady=(2, 12))
        self.grid_columnconfigure(0, weight=1)

    def set_model_note(self, text: str, warn: bool = False) -> None:
        """Dòng chú thích dưới ô chọn mô hình (số lớp, nguồn thứ tự lớp, hoặc cảnh báo)."""
        self._model_note.configure(text=text, text_color=theme.WARN if warn else theme.MUTED)

    def set_model(self, name: str) -> None:
        self._model_menu.set(name)

    def _model_changed(self, name: str) -> None:
        if self._on_model is not None:
            self._on_model(name)

    def _switch(self, text: str, value: bool, row: int, command) -> ctk.CTkSwitch:
        sw = ctk.CTkSwitch(self, text=text, command=command, font=theme.font(12), text_color=theme.TEXT,
                           progress_color=theme.ACCENT, button_color=theme.TEXT,
                           button_hover_color=theme.TEXT, fg_color=theme.PANEL)
        sw.grid(row=row, column=0, columnspan=2, sticky="w", padx=14, pady=2)
        if value:
            sw.select()
        return sw

    def _conf_changed(self, value: float) -> None:
        self._conf_value.configure(text=f"{float(value):.0%}")
        self._on_conf(float(value))

    def _sound_changed(self) -> None:
        self._on_sound(bool(self._sound.get()))

    def _hide_changed(self) -> None:
        if self._on_hide is not None:
            self._on_hide(bool(self._hide.get()))

    def _tta_changed(self) -> None:
        self._on_tta(bool(self._tta.get()))
