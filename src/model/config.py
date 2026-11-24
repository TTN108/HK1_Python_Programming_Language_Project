# src/model/config.py
from typing import Dict, List

# 10 lớp đầu ra cố định: 9 mệnh giá + 1 lớp không phải tiền
CLASS_NAMES: List[str] = [
    "1000",
    "2000",
    "5000",
    "10000",
    "20000",
    "50000",
    "100000",
    "200000",
    "500000",
    "khong_phai_tien",
]

IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

UNCERTAIN_LABEL: str = "không chắc chắn"
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.90
IMAGE_SIZE: int = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)