"""Nắn phẳng tờ tiền theo 4 góc -> ảnh 224x224 RGB, kèm chỉ số chất lượng.

Hợp đồng:
    đầu vào : khung hình BGR + NoteDetection (4 góc theo chiều kim đồng hồ, cạnh TL->TR là cạnh dài)
    đầu ra  : RectifiedNote
              image     ảnh tờ 224x224, RGB, uint8 (đưa thẳng vào bộ phân loại)
              sharpness phương sai Laplacian của ảnh nắn phẳng (càng cao càng nét)
              area_ratio tỉ lệ diện tích tờ trong khung
              occlusion mức bị che ước lượng 0..1 (từ độ tin cậy góc + vùng da quanh tờ)
              reasons   các lý do chất lượng: 'blur' | 'too_small' | 'occluded' | 'cut_by_edge'
                        ('overlap' do pipeline gắn thêm khi hai tờ giao nhau)
              quality   0..1, dùng làm trọng số phiếu khi theo dõi
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app import config
from app.core.note_detector import NoteDetection


@dataclass(frozen=True)
class RectifiedNote:
    image: np.ndarray
    sharpness: float
    area_ratio: float
    occlusion: float
    reasons: tuple[str, ...]
    quality: float


def warp_note(frame_bgr: np.ndarray, corners: np.ndarray, size: int = config.WARP_SIZE) -> np.ndarray:
    """Nắn phẳng theo 4 góc (TL, TR, BR, BL) thành ảnh vuông size x size, vẫn ở thứ tự BGR."""
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    return cv2.warpPerspective(frame_bgr, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sharpness_of(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_occlusion(det: NoteDetection) -> float:
    """Ước lượng mức bị che từ độ tin cậy 4 góc và vùng da quanh tờ (heuristic)."""
    low = float(np.mean(det.corner_conf < 0.9))
    return float(np.clip(0.5 * low + 0.5 * det.hand_contact, 0.0, 1.0))


def rectify(frame_bgr: np.ndarray, det: NoteDetection, size: int = config.WARP_SIZE) -> RectifiedNote:
    h, w = frame_bgr.shape[:2]
    corners = np.asarray(det.corners, np.float32)
    flat = warp_note(frame_bgr, corners, size)
    sharp = sharpness_of(flat)
    area_ratio = float(cv2.contourArea(corners) / float(w * h))
    occlusion = estimate_occlusion(det)

    reasons = []
    margin = max(config.EDGE_MARGIN_PX, 0.005 * w)
    if (corners[:, 0] < margin).any() or (corners[:, 0] > w - 1 - margin).any() \
            or (corners[:, 1] < margin).any() or (corners[:, 1] > h - 1 - margin).any():
        reasons.append("cut_by_edge")
    if area_ratio < config.MIN_AREA_RATIO:
        reasons.append("too_small")
    if sharp < config.MIN_SHARPNESS:
        reasons.append("blur")
    if occlusion > config.MAX_OCCLUSION:
        reasons.append("occluded")

    quality = (
        0.5 * float(np.clip(sharp / (3.0 * config.MIN_SHARPNESS), 0.0, 1.0))
        + 0.3 * (1.0 - occlusion)
        + 0.2 * float(np.clip(area_ratio / 0.05, 0.0, 1.0))
    )
    rgb = np.ascontiguousarray(flat[..., ::-1])
    return RectifiedNote(rgb, sharp, area_ratio, occlusion, tuple(reasons), float(np.clip(quality, 0.0, 1.0)))