"""Toàn bộ hằng số & ngưỡng của ứng dụng nhận diện tiền VND, gộp trong một file.

Các nhóm: đường dẫn & thư mục dữ liệu | camera | tốc độ xử lý | số tờ mỗi lượt | nắn phẳng & chất lượng khung |
bộ phân loại (+ danh sách lớp CLASSES) | theo dõi & chống nhảy số | âm báo.
Ngưỡng là điểm khởi đầu, cần chỉnh lại bằng ảnh thật (tools/evaluate.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # thư mục gốc dự án (chứa main.py, models/)
if getattr(sys, "frozen", False):                   # dữ liệu ghi ra nằm cạnh file .exe khi đã đóng gói
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = BASE_DIR

MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "currency_classifier_v1.onnx"   # mô hình mở mặc định (đổi được trong giao diện)
CLASSES_JSON = MODELS_DIR / "classes.json"  # (tuỳ chọn) thứ tự lớp CHUNG, chỉ dùng khi mô hình không có file lớp riêng
# Mỗi mô hình nên có file lớp riêng cạnh nó: models/<tên>.classes.json (training.export_onnx tự ghi).

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
DETECT_REFINE = True       # tinh chỉnh cạnh ở độ phân giải gốc (góc chính xác hơn, ít rung)
DETECT_TEMPORAL = True     # theo dõi tờ giữa các khung (mượt, nhanh hơn)
DETECT_FULL_EVERY = 3      # khi theo dõi: tìm thô toàn khung mỗi N khung (1 = mọi khung, chậm hơn)
DETECT_WORK_WIDTH = 480     # bộ tìm tờ thu khung về rộng chừng này (640 chính xác hơn chút nhưng chậm hơn ~1,7 lần)
CLASSIFY_EVERY = 2          # chỉ phân loại mỗi N khung (1 = mọi khung); khung bỏ qua vẫn theo dõi vị trí, chỉ không bỏ phiếu
DETECT_EXTRA = 3            # bộ tìm đưa thêm chừng này ứng viên ngoài MAX_NOTES để bộ phân loại LỌC (vật nền bị loại, tờ thật ở lại)
FAST_START_CLASSIFY = 3     # vật mới được phân loại MỌI khung cho tới khi đủ số lần này (hiện kết quả nhanh)

# Lọc vật không phải tiền bằng bộ phân loại (nhận ra màu/hoa văn tiền thật tốt hơn hình học thuần tuý)
HIDE_NOT_MONEY = True       # ẩn vật bị xác nhận "khong_phai_tien" (bật/tắt ngay trong giao diện)
HIDE_NOT_MONEY_CONF = 0.80  # độ tin cậy tối thiểu của kết luận "không phải tiền" để được tính là chắc chắn
NOT_MONEY_CONFIRM = 2       # số lần phân loại liên tiếp chắc chắn là "không phải tiền" mới ẩn (tránh ẩn nhầm tờ thật vì một khung xấu)
REJECT_RECHECK = 4          # vật đã ẩn chỉ được kiểm tra lại mỗi N lượt phân loại (đỡ tốn thời gian)

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
ONNX_THREADS = max(1, min(4, (os.cpu_count() or 4) // 2))   # chừa nhân CPU cho camera/giao diện/bộ tìm tờ; 0 = mặc định của onnxruntime

# Mệnh giá mặc định (đồng). Thêm bớt cho khớp mô hình đã train.
DEFAULT_DENOMINATIONS = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]


def _read_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data["classes"] if isinstance(data, dict) else data
    classes = [str(c) for c in classes]
    if len(set(classes)) != len(classes) or classes.count(NOT_MONEY_CLASS) != 1:
        raise ValueError(f"{path.name}: danh sách lớp phải không trùng và có đúng một lớp '{NOT_MONEY_CLASS}'")
    return classes


def classes_for_model(model_path: Path | str | None = None) -> tuple[list[str], str]:
    """(danh sách lớp, nguồn) cho một mô hình — danh sách PHẢI đúng thứ tự đầu ra của mô hình đó.

    Nguồn: "sidecar" = file riêng models/<tên>.classes.json (đáng tin nhất)
           "shared"  = models/classes.json dùng chung (có thể sai nếu mô hình train theo thứ tự khác)
           "default" = thứ tự chuỗi tăng dần như torchvision ImageFolder ('1000', '10000', '100000', '2000', ...)
    """
    if model_path is not None:
        sidecar = Path(model_path).with_suffix(".classes.json")
        if sidecar.exists():
            return _read_classes(sidecar), "sidecar"
    if CLASSES_JSON.exists():
        return _read_classes(CLASSES_JSON), "shared"
    return sorted(str(d) for d in DEFAULT_DENOMINATIONS) + [NOT_MONEY_CLASS], "default"


def load_classes() -> list[str]:
    """Danh sách lớp của mô hình mặc định (dùng cho ô chọn mệnh giá tay; chỉ tập lớp quan trọng, không phải thứ tự)."""
    return classes_for_model(CLASSIFIER_PATH)[0]


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
MIN_VOTES = 3               # số phiếu tối thiểu để chốt nhãn (đồng thuận vẫn phải >= CONSENSUS_RATIO)
CONSENSUS_RATIO = 0.60      # tỉ lệ đồng thuận tối thiểu
MAX_MISSED_FRAMES = 8       # mất tờ tạm thời tới số khung này vẫn giữ track (tổng giữ nguyên)
TRACK_MIN_IOU = 0.20        # ghép track theo IoU tối thiểu
TRACK_MAX_CENTER_DIST = 0.60  # hoặc khoảng cách tâm tối đa (tỉ lệ so với cạnh dài của tờ)
VOTE_WEIGHT_MIN = 0.20      # trọng số phiếu thấp nhất
TOTAL_STABLE_FRAMES = 12    # tổng giữ nguyên số khung này -> "đã ổn định"
RELEASE_FRAMES = 15         # không thấy tờ nào số khung này -> được ghi lượt mới
NEW_TRACK_MIN_AGE = 10      # tờ mới phải sống số khung này mới tính là "tập tờ đã đổi"

# ----------------------------------------------------------------------------
# Âm báo
# ----------------------------------------------------------------------------
SOUND_ENABLED = True