"""Toàn bộ hằng số & ngưỡng của ứng dụng nhận diện tiền VND, gộp trong một file.

Các nhóm: đường dẫn & thư mục dữ liệu | camera | tốc độ xử lý | số tờ mỗi lượt | nắn phẳng & chất lượng khung |
bộ phân loại (+ danh sách lớp CLASSES) | theo dõi & chống nhảy số | âm báo.
Ngưỡng là điểm khởi đầu, cần chỉnh lại bằng ảnh thật (tools/evaluate.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # nơi chứa mã nguồn và thư mục models/
if getattr(sys, "frozen", False):                   # dữ liệu ghi ra nằm cạnh file .exe khi đã đóng gói
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = BASE_DIR

MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "currency_classifier_v1.onnx"
CLASSES_JSON = MODELS_DIR / "classes.json"  # (tuỳ chọn) mảng tên lớp đúng thứ tự lúc train

OUTPUT_DIR = APP_DIR / "outputs"
DB_PATH = OUTPUT_DIR / "kiem_dem.db"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
EXPORT_DIR = OUTPUT_DIR / "exports"

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
CAMERA_SOURCE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
LOCK_EXPOSURE = False       # khoá phơi sáng / cân bằng trắng (tuỳ hệ điều hành, bật bằng --lock-exposure)
FOCUS_VALUE = None          # số = tắt autofocus và khoá nét ở giá trị này (tuỳ driver; None = để camera tự lấy nét)
EXPOSURE_VALUE = None       # số = tắt tự phơi sáng và đặt thời gian phơi sáng (DirectShow thường là số âm, vd -6)

# ----------------------------------------------------------------------------
# Tốc độ xử lý
# ----------------------------------------------------------------------------
DETECT_WORK_WIDTH = 480     # bộ tìm tờ thu khung về rộng chừng này (640 chính xác hơn chút nhưng chậm hơn ~1,7 lần)
CLASSIFY_EVERY = 2          # chỉ phân loại mỗi N khung (1 = mọi khung); khung bỏ qua vẫn theo dõi vị trí, chỉ không bỏ phiếu

# ----------------------------------------------------------------------------
# Số tờ mỗi lượt
# ----------------------------------------------------------------------------
MAX_NOTES = 5

# ----------------------------------------------------------------------------
# Nắn phẳng & chất lượng khung
# ----------------------------------------------------------------------------
MIN_AREA_RATIO = 0      # tờ chiếm ít hơn tỉ lệ này của khung -> "too_small" (không phân loại)
MIN_SHARPNESS = 20.0        # phương sai Laplacian của ảnh nắn phẳng; thấp hơn -> "blur"
MAX_OCCLUSION = 10       # mức bị che ước lượng; cao hơn -> "occluded"
EDGE_MARGIN_PX = 0          # góc cách mép khung ít hơn số này -> "cut_by_edge"
OVERLAP_RATIO = 0.05        # hai tờ giao nhau quá tỉ lệ này (so với tờ nhỏ hơn) -> "overlap"

# Các lý do chất lượng, chia hai nhóm:
WARNING_REASONS = ("overlap", "cut_by_edge", "occluded")   # cảnh báo: chặn xác nhận, không cộng tiền
NO_VOTE_REASONS = WARNING_REASONS + ("blur", "too_small")  # khung có lý do này không được bỏ phiếu

# ----------------------------------------------------------------------------
# Bộ phân loại
# ----------------------------------------------------------------------------
WARP_SIZE = 224                 # ảnh tờ đưa vào bộ phân loại: 224x224, RGB
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NOT_MONEY_CLASS = "khong_phai_tien"
CONF_THRESHOLD = 0.70           # dưới ngưỡng -> status "uncertain" (tham số --conf)
USE_TTA = True                  # chạy thêm lượt xoay 180° cho tờ kém chắc chắn
TTA_SKIP_CONF = 0.90            # kết quả đầu >= ngưỡng này thì bỏ qua lượt xoay (nhanh gần gấp đôi)
ONNX_THREADS = 0                # số luồng onnxruntime; 0 = mặc định

# Mệnh giá mặc định (đồng). Thêm bớt cho khớp mô hình đã train.
DEFAULT_DENOMINATIONS = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]


def load_classes() -> list[str]:
    """Danh sách lớp, PHẢI đúng thứ tự đầu ra của bộ phân loại.

    Ưu tiên đọc models/classes.json (mảng tên lớp theo thứ tự lúc train).
    Nếu chưa có, dùng thứ tự chuỗi tăng dần giống torchvision ImageFolder
    ('1000', '10000', '100000', '2000', ...) rồi tới lớp không phải tiền.
    Nếu mô hình train theo thứ tự khác, hãy tạo classes.json.
    """
    if CLASSES_JSON.exists():
        return json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
    return sorted(str(d) for d in DEFAULT_DENOMINATIONS) + [NOT_MONEY_CLASS]


CLASSES = load_classes()


def class_to_value(name: str) -> int:
    """Tên lớp -> số tiền (đồng). Lớp không phải tiền = 0.

    Hiểu các dạng '50000', '50.000', '50k', '500K'.
    """
    s = name.strip().lower().replace(".", "").replace(",", "").replace("_", "")
    if s.endswith("k") and s[:-1].isdigit():
        return int(s[:-1]) * 1000
    return int(s) if s.isdigit() else 0


# ----------------------------------------------------------------------------
# Theo dõi (track) & chống nhảy số
# ----------------------------------------------------------------------------
HISTORY_LEN = 7             # số khung dùng để bỏ phiếu
MIN_VOTES = 4               # số phiếu tối thiểu để chốt nhãn
CONSENSUS_RATIO = 0.60      # tỉ lệ đồng thuận tối thiểu
MAX_MISSED_FRAMES = 8       # mất tờ tạm thời tới số khung này vẫn giữ track (tổng giữ nguyên)
TRACK_MIN_IOU = 0.20        # ghép track theo IoU tối thiểu
TRACK_MAX_CENTER_DIST = 0.60  # hoặc khoảng cách tâm tối đa (tỉ lệ so với cạnh dài của tờ)
VOTE_WEIGHT_MIN = 0.20      # trọng số phiếu thấp nhất
TOTAL_STABLE_FRAMES = 20    # tổng giữ nguyên số khung này -> "đã ổn định"
RELEASE_FRAMES = 15         # không thấy tờ nào số khung này -> được ghi lượt mới
NEW_TRACK_MIN_AGE = 10      # tờ mới phải sống số khung này mới tính là "tập tờ đã đổi"

# ----------------------------------------------------------------------------
# Âm báo
# ----------------------------------------------------------------------------
SOUND_ENABLED = True