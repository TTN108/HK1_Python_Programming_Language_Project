"""Lưu trữ: lịch sử các lượt đếm (xuất CSV) và ảnh chụp. Thư mục đích lấy từ app/config.py."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app import config


@dataclass(frozen=True)
class SessionEntry:
    time: str
    n_notes: int
    total: int
    breakdown: str          # ví dụ "2x50000; 1x20000"


class SessionLog:
    """Danh sách các lượt đã bấm 'Ghi nhận'."""

    def __init__(self) -> None:
        self.entries: list[SessionEntry] = []

    def add(self, total: int, counts: dict[int, int]) -> SessionEntry:
        breakdown = "; ".join(f"{n}x{value}" for value, n in sorted(counts.items(), reverse=True))
        entry = SessionEntry(time.strftime("%H:%M:%S"), sum(counts.values()), total, breakdown)
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries.clear()

    @property
    def grand_total(self) -> int:
        return sum(e.total for e in self.entries)

    def export_csv(self, path: str | Path | None = None) -> Path:
        if path is None:
            config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            path = config.EXPORT_DIR / time.strftime("kiem_dem_%Y%m%d_%H%M%S.csv")
        path = Path(path)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig để Excel đọc đúng tiếng Việt
            f.write("luot,thoi_gian,so_to,tong_tien,chi_tiet\n")
            for i, e in enumerate(self.entries, 1):
                f.write(f"{i},{e.time},{e.n_notes},{e.total},\"{e.breakdown}\"\n")
            f.write(f",,,{self.grand_total},TONG CONG\n")
        return path


def save_snapshot(frame_bgr: np.ndarray) -> Path:
    """Lưu khung hình hiện tại vào outputs/snapshots/ (đọc/ghi được cả đường dẫn có dấu)."""
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SNAPSHOT_DIR / time.strftime("snap_%Y%m%d_%H%M%S.jpg")
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Không mã hoá được ảnh")
    buf.tofile(str(path))
    return path
