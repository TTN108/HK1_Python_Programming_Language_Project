"""Mọi chữ hiển thị trên giao diện + hàm định dạng tiền. MUỐN ĐỔI CÂU CHỮ -> SỬA FILE NÀY."""
from __future__ import annotations

APP_TITLE = "Trạm quyên góp minh bạch"
APP_SUBTITLE = "Nhận diện mệnh giá tức thì · tổng cập nhật theo thời gian thực"
WINDOW_TITLE = "Trạm quyên góp thiện nguyện minh bạch · Computer Vision"

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
SETTING_SOUND = "Âm thanh cảm ơn"
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

# --- trạm quyên góp (mới) ---
TOTAL_TITLE_KIOSK = "TỔNG QUYÊN GÓP"
HINT_NO_SLOT = "⚠ Chưa chọn khe bỏ tiền — chưa cộng tiền"
HINT_KIOSK_EMPTY = "Đưa tờ tiền trước camera khoảng 1 giây, rồi bỏ vào khe"
HINT_AWAIT_SLOT = "Đã nhận diện ✓  Hãy bỏ tờ tiền vào khe"
DONATED_TEXT = "Đã bỏ vào khe ✓"
BTN_SLOT = "Chọn khe"
BTN_SLOT_LOCKED = "Khe đã khoá ✓"
BTN_DASHBOARD = "Màn hình công khai"
SLOT_PICK_HELP = "Bấm 4 góc của khe bỏ tiền trên video (chuột phải: bỏ điểm vừa chọn)"
SLOT_CONFIRM_TITLE = "Khoá vùng khe"
SLOT_CONFIRM = "Khoá vùng khe này?\nMuốn đổi phải chạy lại chương trình."
SLOT_LOCKED = "Đã khoá vùng khe. Từ giờ chỉ tờ bỏ vào khe mới được cộng tiền"
SLOT_LOCKED_INFO = "Vùng khe đã khoá; muốn đổi hãy chạy lại chương trình"
SLOT_NEED_VIDEO = "Hãy bấm Bắt đầu để có hình trước khi chọn khe"
SLOT_SIZE_CHANGED = "Độ phân giải khung hình đã đổi, vùng khe không còn khớp — hãy chạy lại chương trình"
RESET_DONE = "Đã làm mới theo dõi tờ (tổng quyên góp giữ nguyên)"
DONATION_FLASH = "Lượt #{seq}: +{value} · tổng {total}"
DONATION_REVIEW = "Lượt #{seq}: tờ vào khe nhưng chưa nhận diện được — cần kiểm tra ảnh nhật ký"
DASH_WINDOW_TITLE = "Quyên góp minh bạch · màn hình công khai"
DASH_TITLE = "Quyên góp minh bạch"
DASH_SUBTITLE = "Cập nhật theo thời gian thực · mỗi lượt đều có nhật ký và ảnh chụp"
DASH_COUNT = "{n} lượt đóng góp"
DASH_NOT_READY = "Trạm đang chuẩn bị…"
DASH_PENDING = "đang xác nhận"
DASH_THANKS = "Cảm ơn bạn! +{value}"
DASH_THANKS_PENDING = "Cảm ơn bạn đã đóng góp!"

# ----------------------------------------------------------------------------
# Định dạng
# ----------------------------------------------------------------------------
def fmt_number(value: int) -> str:
    """50000 -> '50.000'"""
    return f"{value:,}".replace(",", ".")


def fmt_vnd(value: int) -> str:
    """50000 -> '50.000 ₫'"""
    return f"{fmt_number(value)} ₫"
