"""Huấn luyện & đánh giá mô hình phân loại mệnh giá (phần của thành viên TV2).

Chạy từ thư mục gốc dự án, theo dạng module:
    python -m training.prepare_data      # tải + cắt dữ liệu -> data_v1/
    python -m training.train             # huấn luyện -> checkpoints/*.pth
    python -m training.export_onnx --checkpoint checkpoints/xxx.pth   # -> models/*.onnx + *.classes.json
    python -m training.evaluate          # báo cáo + confusion matrix

Ứng dụng (app/) KHÔNG import gói này; hai bên chỉ gặp nhau qua file trong models/.
Chi tiết: docs/TRAINING.md
"""
