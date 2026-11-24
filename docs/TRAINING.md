# Huấn luyện mô hình phân loại (gói `training/`, phần của TV2)

Chạy **từ thư mục gốc dự án**, dạng module (`python -m training.xxx`). Cài thư viện: `pip install -r requirements-train.txt`.

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Dữ liệu | `ROBOFLOW_API_KEY=... python -m training.prepare_data` (có `--skip_download` nếu đã tải) | `data_v1/{train,val,test}/<lớp>/*.jpg` (crop 224×224 từ nhãn YOLO + ảnh nền `khong_phai_tien`) |
| 2. Huấn luyện | `python -m training.train --backbone mobilenet_v3_small` | `checkpoints/best_*.pth`, `logs/experiments_log.csv` |
| 3. Xuất ONNX | `python -m training.export_onnx --checkpoint checkpoints/best_....pth` | `models/vnd_classifier_v1.onnx` **và** `models/vnd_classifier_v1.classes.json` |
| 4. Đánh giá | `python -m training.evaluate --split_dir data_v1/test` | báo cáo, `logs/confusion_matrix.png`, khảo sát ngưỡng tin cậy |

Sau bước 3 chỉ cần mở app: mô hình mới tự hiện trong ô **Mô hình** (cài đặt nhanh, cột phải).

## Hợp đồng với ứng dụng

- Hai bên chỉ gặp nhau qua `models/`; `app/` không import `training/`.
- Thứ tự lớp lúc train = `CLASS_NAMES` trong `training/config.py`: `1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, khong_phai_tien`. File `.classes.json` ghi đúng thứ tự này. **Đổi `CLASS_NAMES` thì phải xuất lại ONNX.**
- Đầu vào mô hình: RGB, 224×224, /255 rồi chuẩn hoá ImageNet (khớp `app/core/classifier.py`). Đầu ra: logits 10 lớp (app tự softmax).
- `training/classifier.py` (`BillClassifier`) chỉ phục vụ `evaluate.py`; khi chạy thật app dùng `app/core/classifier.py` (có TTA, trạng thái ok/uncertain/not_money).
- Ngưỡng tin cậy: `evaluate` khảo sát 0,80–0,95 và `training` mặc định 0,90; app mặc định 0,70 (`CONF_THRESHOLD`, chỉnh bằng thanh trượt). Chọn lại theo kết quả khảo sát của mô hình mới.

## Cần kiểm tra bằng ảnh thật

- Dữ liệu train là ảnh crop theo khung YOLO (không nắn phẳng), còn app đưa vào ảnh đã **nắn phẳng theo 4 góc** (`app/core/rectify.py`, cạnh dài nằm ngang). Nếu độ chính xác ngoài đời thấp hơn `evaluate`, hãy thêm vào tập train các ảnh đã qua `rectify` từ camera thật.
- `albumentations>=2.0` là bắt buộc (`CoarseDropout(num_holes_range, fill)` không có ở 1.x).
