"""Màu sắc, phông chữ, kích thước chung. MUỐN ĐỔI DIỆN MẠO CẢ ỨNG DỤNG -> SỬA FILE NÀY.

Chủ đề tối xanh navy. Mọi widget đều lấy màu từ đây, không viết mã màu trực tiếp trong widget.
"""
from __future__ import annotations

import sys

# ----------------------------------------------------------------------------
# Màu nền & chữ
# ----------------------------------------------------------------------------
BG = "#0F1220"            # nền cửa sổ
PANEL = "#171B2E"         # khung lớn (video, cột kết quả)
CARD = "#1F243B"          # thẻ nhỏ bên trong khung
CARD_HOVER = "#2A3150"
VIDEO_BG = "#0B0E1A"
BORDER = "#2B3252"

TEXT = "#E8ECFF"
MUTED = "#8A93B8"
ON_ACCENT = "#FFFFFF"

ACCENT = "#6C8CFF"        # màu nhấn chính (xanh tím)
ACCENT_HOVER = "#5877F0"
GOOD = "#35D6A4"          # thành công / đã ổn định
GOOD_HOVER = "#2BBF91"
WARN = "#F5B94A"          # chưa chắc chắn
DANGER = "#FF6B7A"        # cảnh báo / dừng
DANGER_HOVER = "#E85666"

# Màu theo trạng thái của từng tờ (khớp với STATUS_* trong app/core/aggregator.py)
STATUS_COLOR = {
    "stable": GOOD,
    "manual": ACCENT,
    "uncertain": WARN,
    "warning": DANGER,
    "not_money": MUTED,
}

# Màu gợi nhớ cho từng mệnh giá (đồng) — dùng cho thanh màu bên trái mỗi dòng
DENOM_COLOR = {
    1000: "#9AA5C9", 2000: "#7FA6C9", 5000: "#5B8DEF",
    10000: "#E6B94D", 20000: "#4FA3E0", 50000: "#F078B0",
    100000: "#4CC38A", 200000: "#E4775A", 500000: "#38C9D6",
}

# ----------------------------------------------------------------------------
# Kích thước & phông
# ----------------------------------------------------------------------------
WINDOW_SIZE = "1280x780"
WINDOW_MIN = (1120, 700)
SIDEBAR_WIDTH = 400
RADIUS = 18               # bo góc khung lớn
RADIUS_SM = 12            # bo góc nút / thẻ nhỏ

if sys.platform.startswith("win"):
    FONT_FAMILY = "Segoe UI"
elif sys.platform == "darwin":
    FONT_FAMILY = "Helvetica Neue"
else:
    FONT_FAMILY = "DejaVu Sans"


def font(size: int = 13, bold: bool = False):
    """Tạo phông. Chỉ gọi SAU khi cửa sổ gốc đã được tạo (widget nào cũng thoả điều này)."""
    import customtkinter as ctk
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight="bold" if bold else "normal")


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (B, G, R) để vẽ bằng OpenCV."""
    c = color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def apply_global_theme() -> None:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
