# src/storage/totals.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional

VALID_DENOMINATIONS: List[int] = [
    1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000
]
VALID_DENOM_STRS = {str(d) for d in VALID_DENOMINATIONS}
UNCERTAIN_LABELS = {"không chắc chắn", "khong_chac_chan"}


@dataclass
class NoteRecord:
    """Cấu trúc 1 tờ tiền theo hợp đồng module giữa TV1, TV2 và TV4."""
    denomination: str                  # VD: "50000", "không chắc chắn", "khong_phai_tien"
    confidence: float                  # Độ tin cậy từ 0.0 đến 1.0
    is_manual_edit: bool = False       # Có được người dùng sửa tay hay không
    bbox: Optional[List[int]] = None   # Tọa độ khung (4 góc hoặc [x1, y1, x2, y2])


@dataclass
class BatchSummary:
    total_amount: int
    valid_note_count: int
    uncertain_count: int
    counts_by_denom: Dict[int, int] = field(default_factory=dict)
    amount_by_denom: Dict[int, int] = field(default_factory=dict)
    is_final: bool = True
    display_label: str = "Tổng"


def calculate_batch_totals(notes: List[NoteRecord]) -> BatchSummary:
    """
    Tính tổng tiền và thống kê số tờ theo từng mệnh giá cho 1 lượt kiểm đếm.
    """
    counts = {d: 0 for d in VALID_DENOMINATIONS}
    amounts = {d: 0 for d in VALID_DENOMINATIONS}
    total_amount = 0
    valid_count = 0
    uncertain_count = 0

    for note in notes:
        denom_str = str(note.denomination).strip()
        if denom_str in VALID_DENOM_STRS:
            val = int(denom_str)
            counts[val] += 1
            amounts[val] += val
            total_amount += val
            valid_count += 1
        elif denom_str.lower() in UNCERTAIN_LABELS:
            uncertain_count += 1
        # Các nhãn khác như "khong_phai_tien" hoặc lỗi khung đỏ sẽ bị bỏ qua

    is_final = uncertain_count == 0
    display_label = (
        "Tổng"
        if is_final
        else f"Tổng tạm tính (còn {uncertain_count} tờ chưa xác nhận)"
    )

    return BatchSummary(
        total_amount=total_amount,
        valid_note_count=valid_count,
        uncertain_count=uncertain_count,
        counts_by_denom=counts,
        amount_by_denom=amounts,
        is_final=is_final,
        display_label=display_label,
    )