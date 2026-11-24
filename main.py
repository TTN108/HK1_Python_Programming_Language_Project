"""Điểm vào của ứng dụng.

    python main.py                       # mở giao diện
    python main.py --source 0            # tự bật webcam số 0
    python main.py --source video.mp4    # tự phát video
    python main.py --no-sound            # tắt đọc to
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Money Counter - kiểm đếm tiền VND bằng camera")
    ap.add_argument("--source", default=None, help="số webcam (0,1,...) hoặc đường dẫn video để tự bắt đầu")
    ap.add_argument("--no-sound", action="store_true", help="tắt đọc to tổng tiền")
    args = ap.parse_args()

    try:
        import customtkinter  # noqa: F401
    except ImportError:
        print("Thiếu thư viện giao diện. Chạy:  pip install -r requirements.txt")
        return 1

    from app.ui.app import run
    run(source=args.source, sound=not args.no_sound)
    return 0


if __name__ == "__main__":
    sys.exit(main())
