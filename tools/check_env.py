"""Kiểm tra môi trường trước khi chạy ứng dụng:  python tools/check_env.py

In phiên bản các thư viện, kiểm tra file mô hình, thử nạp mô hình và (tuỳ chọn) thử giọng đọc.
Thay cho test_doan.py cũ (file đó kiểm tra torch/ultralytics, chỉ cần khi huấn luyện).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("--- KIỂM TRA MÔI TRƯỜNG ---")
print(f"Python {sys.version.split()[0]}  (khuyến nghị 3.12, tránh 3.14)")
ok = True
for mod, pip_name in [("cv2", "opencv-python"), ("numpy", "numpy"), ("PIL", "pillow"),
                      ("onnxruntime", "onnxruntime"), ("customtkinter", "customtkinter"),
                      ("pyttsx3", "pyttsx3 (tuỳ chọn)")]:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK]   {mod:14s} {getattr(m, '__version__', '')}")
    except Exception:  # noqa: BLE001
        optional = "tuỳ chọn" in pip_name
        print(f"  [{'--' if optional else 'THIẾU'}] {mod:14s} -> pip install {pip_name.split()[0]}")
        ok = ok and optional

try:
    from app import config
    from app.core.classifier import CurrencyClassifier
    print(f"Mô hình: {config.CLASSIFIER_PATH}")
    clf = CurrencyClassifier()
    clf.warmup()
    print(f"  [OK]   nạp được mô hình, {len(clf.classes)} lớp: {clf.classes}")
except Exception as exc:  # noqa: BLE001
    ok = False
    print(f"  [LỖI]  {type(exc).__name__}: {exc}")

if "--tts" in sys.argv:
    try:
        import pyttsx3
        e = pyttsx3.init()
        e.say("Môi trường nhận diện mệnh giá tiền đã sẵn sàng")
        e.runAndWait()
    except Exception as exc:  # noqa: BLE001
        print(f"  [LỖI]  giọng đọc: {exc}")

print("\nSẴN SÀNG ✔  chạy:  python main.py" if ok else "\nCòn lỗi ở trên — sửa rồi chạy lại.")
