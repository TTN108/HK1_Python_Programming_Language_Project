"""Mọi chữ hiển thị trên giao diện + hàm định dạng tiền. MUỐN ĐỔI CÂU CHỮ -> SỬA FILE NÀY."""
from __future__ import annotations

APP_TITLE = "Money Counter"
APP_SUBTITLE = "Kiểm đếm tiền VND bằng Computer Vision"
WINDOW_TITLE = "Money Counter · Kiểm đếm tiền VND bằng Computer Vision"

# --- video -----------------------------------------------------------------
VIDEO_PLACEHOLDER = "Nhấn  ▶ Bắt đầu  để mở camera\nhoặc  Mở tệp…  để thử với ảnh / video"

# --- thẻ tổng tiền ------------------------------------------------------------
TOTAL_TITLE = "TỔNG TIỀN"
HINT_EMPTY = "Đặt tiền vào khung hình, các tờ không chồng lên nhau"
HINT_WORKING = "Đang nhận diện…"
HINT_NOT_MONEY = "Vật trong khung hình không được nhận là tiền"
HINT_STABLE = "Đã ổn định ✓  Có thể ghi nhận"
HINT_WARNING = "⚠ Có tờ bị chồng / sát mép / bị che — hãy tách ra"

# --- danh sách tờ ---------------------------------------------------------------
LIST_TITLE = "Chi tiết từng tờ"
LIST_EMPTY = "Chưa phát hiện tờ tiền nào"
AUTO_LABEL = "Tự động"
NOT_MONEY_LABEL = "Không phải tiền"
LOST_SUFFIX = " · mất dấu"

STATUS_TEXT = {
    "stable": "Đã xác nhận",
    "manual": "Sửa tay",
    "uncertain": "Chưa chắc chắn",
    "warning": "Cảnh báo",
    "not_money": "Không phải tiền",
}
REASON_TEXT = {
    "overlap": "chồng tờ",
    "cut_by_edge": "sát mép",
    "occluded": "bị che",
    "blur": "mờ",
    "too_small": "quá nhỏ",
}

# --- nút -------------------------------------------------------------------------
BTN_START = "▶  Bắt đầu"
BTN_STOP = "■  Dừng"
BTN_OPEN = "Mở tệp…"
BTN_SNAPSHOT = "Chụp ảnh"
BTN_COMMIT = "✔  Ghi nhận lượt"
BTN_RESET = "↺  Làm lại"
BTN_EXPORT = "⬇  Xuất CSV"

# --- cài đặt nhanh ------------------------------------------------------------------
SETTINGS_TITLE = "Cài đặt nhanh"
SETTING_MODEL = "Mô hình nhận diện"
SETTING_CONF = "Ngưỡng tin cậy"
SETTING_SOUND = "Đọc to khi ổn định"
SETTING_TTA = "TTA (xoay 180° cho tờ khó)"
SETTING_HIDE = "Ẩn vật không phải tiền"
STATUS_HIDDEN = "ẩn {n} vật không phải tiền"

# --- trạng thái mô hình ------------------------------------------------------------------
MODEL_OK = "Mô hình sẵn sàng"
MODEL_ERROR = "Chưa nạp được mô hình"
MODEL_INFO = "{n} lớp · {src} · {mb:.1f} MB"
MODEL_UNTRUSTED = ("⚠ Thiếu file {name}.classes.json: thứ tự lớp chỉ là suy đoán, "
                   "có thể nhận nhầm mệnh giá. Chạy lại export_onnx hoặc tạo file này.")
MODEL_NONE = "Không tìm thấy file .onnx nào trong models/"
MODEL_SWITCHED = "Đã đổi mô hình: {name} (lượt đếm được làm lại)"
MODEL_SWITCH_FAILED = "Không đổi được mô hình: {err}"


# ----------------------------------------------------------------------------
# Định dạng
# ----------------------------------------------------------------------------
def fmt_number(value: int) -> str:
    """50000 -> '50.000'"""
    return f"{value:,}".replace(",", ".")


def fmt_vnd(value: int) -> str:
    """50000 -> '50.000 ₫'"""
    return f"{fmt_number(value)} ₫"
