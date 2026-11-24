"""Chạy bộ tìm + lọc trên ẢNH/VIDEO của bạn và vẽ MỌI ứng viên để xem vì sao tờ tiền bị bỏ lỡ hoặc vật nền bị nhận nhầm.

    python tools/debug_detect.py anh_cam_tay.jpg                 # -> anh_cam_tay.debug.png
    python tools/debug_detect.py video.mp4 --frames 30           # lưu mỗi khung thứ 5 trong 30 khung đầu

Màu: XANH = giữ (đưa vào bộ đếm) · ĐỎ = bị ẩn vì bộ phân loại xác nhận không phải tiền. Nhãn: điểm hình học / kết luận bộ phân loại.
Gửi ảnh .debug.png này cho người chỉnh bộ tìm để biết chính xác chỗ sai.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.classifier import CurrencyClassifier  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402


def draw(frame, pipe, timings):
    out = frame.copy()
    for d in pipe.debug:
        col = (0, 0, 255) if d["hidden"] else (0, 200, 0)
        cv2.polylines(out, [d["corners"].astype(np.int32)], True, col, 3, cv2.LINE_AA)
        x, y = d["corners"][0]
        tag = f"{d['score']:.2f} {d['label'] or '?'}" + (f" {d['conf']:.0%}" if d["conf"] else "")
        cv2.putText(out, tag, (int(x), max(18, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(out, f"tim {timings['detect_ms']:.0f}ms  phan loai {timings['classify_ms']:.0f}ms  an {timings['hidden']}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()
    pipe = Pipeline(CurrencyClassifier())
    src = Path(a.source)
    img = cv2.imread(str(src)) if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else None
    if img is not None:
        for _ in range(4):                       # lặp vài lần cho bộ lọc/theo dõi ổn định
            _, tm = pipe.process(img)
        cv2.imwrite(str(src.with_suffix(".debug.png")), draw(img, pipe, tm))
        print("đã lưu", src.with_suffix(".debug.png"))
        return
    cap = cv2.VideoCapture(str(src))
    for i in range(a.frames):
        ok, f = cap.read()
        if not ok:
            break
        _, tm = pipe.process(f)
        if i % a.every == 0:
            out = src.with_name(f"{src.stem}.debug{i:03d}.png")
            cv2.imwrite(str(out), draw(f, pipe, tm))
            print("đã lưu", out)


if __name__ == "__main__":
    main()
