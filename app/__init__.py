"""Ứng dụng kiểm đếm tiền VND bằng camera.

Cấu trúc (xem README.md ở thư mục gốc):
    app/config.py      mọi hằng số & ngưỡng
    app/core/          lõi xử lý ảnh (camera, tìm tờ, nắn phẳng, phân loại, tổng hợp)
    app/engine.py      cầu nối: chạy lõi ở luồng nền, đưa kết quả cho giao diện
    app/services/      dịch vụ phụ: đọc to (TTS), lưu lịch sử/ảnh chụp
    app/ui/            giao diện (theme, widgets, cửa sổ chính)
"""
