"""Vẽ khung + nhãn từng tờ lên khung hình camera (OpenCV).

cv2.putText KHÔNG vẽ được tiếng Việt có dấu nên nhãn trên ảnh chỉ dùng số và chữ không dấu;
chữ tiếng Việt đầy đủ nằm ở danh sách bên phải (app/ui/widgets/note_list.py).
Màu khung theo trạng thái lấy từ app/ui/theme.py (STATUS_COLOR).
"""
from __future__ import annotations

import cv2
import numpy as np

from app import config
from app.core.aggregator import COUNTED, FrameState, TrackView
from app.ui import theme
from app.ui.texts import fmt_number

_REASON_ASCII = {
    "overlap": "chong to",
    "cut_by_edge": "sat mep",
    "occluded": "bi che",
    "blur": "mo",
    "too_small": "qua nho",
}


def _label_text(v: TrackView) -> str:
    if v.donated:                                        # mới
        return f"{fmt_number(v.value)} vao khe"
    if v.status in COUNTED:
        return f"{fmt_number(v.value)}"
    if v.status == "not_money":
        return "khong phai tien"
    if v.status == "warning":
        reasons = [r for r in v.reasons if r in config.WARNING_REASONS]
        return "! " + "/".join(_REASON_ASCII.get(r, r) for r in reasons)
    guess = v.guess
    if guess and guess != config.NOT_MONEY_CLASS:
        return f"{fmt_number(config.class_to_value(guess))}?"
    return "?"


def draw_tracks(frame_bgr: np.ndarray, state: FrameState) -> np.ndarray:
    """Trả về BẢN SAO của khung hình đã vẽ kết quả (không sửa khung gốc)."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    thick = max(2, w // 420)
    scale = max(0.55, w / 1500)
    for v in state.tracks:
        color = theme.hex_to_bgr(theme.STATUS_COLOR.get(v.status, theme.ACCENT))
        pts = np.round(v.corners).astype(np.int32)
        cv2.polylines(out, [pts], True, color, thick if v.visible else 1, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), thick + 2, color, -1, cv2.LINE_AA)

        text = f"#{v.id}  {_label_text(v)}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x0 = int(np.clip(pts[:, 0].min(), 0, max(w - tw - 12, 0)))
        y0 = int(pts[:, 1].min()) - 6
        if y0 - th - 10 < 0:                       # sát mép trên -> đặt nhãn xuống dưới tờ
            y0 = int(pts[:, 1].max()) + th + 14
        cv2.rectangle(out, (x0, y0 - th - 8), (x0 + tw + 12, y0 + base), color, -1)
        cv2.putText(out, text, (x0 + 6, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (20, 20, 20), 2, cv2.LINE_AA)
    return out

def draw_slot(img: np.ndarray, pts, locked: bool) -> np.ndarray:        # mới
    """Vẽ vùng khe bỏ tiền lên khung đã vẽ (sửa thẳng img). locked=False: đang chọn, hiện các điểm đã bấm."""
    if pts is None or len(pts) == 0:
        return img
    p = np.round(np.asarray(pts, np.float32)).astype(np.int32).reshape(-1, 1, 2)
    color = theme.hex_to_bgr(theme.GOOD if locked else theme.WARN)
    thick = max(2, img.shape[1] // 420)
    closed = locked or len(p) >= 4
    if closed and len(p) >= 3:
        shade = img.copy()
        cv2.fillPoly(shade, [p], color)
        cv2.addWeighted(shade, 0.25, img, 0.75, 0, dst=img)
    cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA)
    for x, y in p.reshape(-1, 2):
        cv2.circle(img, (int(x), int(y)), thick + 3, color, -1, cv2.LINE_AA)
    x0, y0 = int(p[:, 0, 0].min()), int(p[:, 0, 1].min())
    cv2.putText(img, "KHE" if locked else f"chon khe {len(p)}/4", (x0, max(y0 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img