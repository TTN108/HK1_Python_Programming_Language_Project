# Money Counter — Kiểm đếm tiền VND bằng Computer Vision

Camera → tìm từng tờ tiền → nắn phẳng → phân loại mệnh giá (MobileNetV3-Small, ONNX) → bỏ phiếu qua nhiều khung hình → **tổng tiền ổn định**, hiển thị trên giao diện, có đọc to bằng tiếng Việt.

## Chạy nhanh

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python tools/check_env.py             # kiểm tra môi trường + nạp thử mô hình
python main.py                        # mở giao diện
```

Tuỳ chọn: `python main.py --source 0` (tự bật webcam), `--source video.mp4`, `--no-sound`.
Yêu cầu Python 3.10+ (khuyến nghị 3.12; **không dùng 3.14**).

## Cấu trúc thư mục

```
money-counter/
├── main.py                  Điểm vào: python main.py
├── requirements.txt         Thư viện để CHẠY app
├── requirements-train.txt   Thư viện để HUẤN LUYỆN (torch, ...) — không cần khi chỉ chạy app
│
├── app/
│   ├── config.py            ★ Mọi hằng số & ngưỡng (camera, độ tin cậy, bỏ phiếu, đường dẫn...)
│   ├── engine.py            Cầu nối: chạy lõi ở luồng nền, đưa kết quả cho giao diện
│   │
│   ├── core/                LÕI XỬ LÝ ẢNH — không biết gì về giao diện
│   │   ├── camera.py          đọc webcam / video ở luồng nền
│   │   ├── note_detector.py   tìm tờ tiền + 4 góc (OpenCV cổ điển)
│   │   ├── rectify.py         nắn phẳng 224×224, đo độ nét/che
│   │   ├── classifier.py      ONNX Runtime: mệnh giá + độ tin cậy
│   │   ├── model_registry.py  liệt kê các mô hình .onnx trong models/ + thứ tự lớp của từng mô hình
│   │   ├── pipeline.py        ghép: tìm → nắn → phân loại
│   │   └── aggregator.py      theo dõi từng tờ, bỏ phiếu, tính tổng, sửa tay
│   │
│   ├── services/            Dịch vụ phụ, độc lập với giao diện
│   │   ├── speech.py          đọc to tổng tiền (pyttsx3, tiếng Việt)
│   │   └── storage.py         lịch sử các lượt (xuất CSV) + lưu ảnh chụp
│   │
│   └── ui/                  GIAO DIỆN (CustomTkinter)
│       ├── theme.py           ★ màu sắc, phông, kích thước — đổi diện mạo ở đây
│       ├── texts.py           ★ mọi câu chữ hiển thị + hàm định dạng tiền
│       ├── overlay.py         vẽ khung/nhãn từng tờ lên video
│       ├── app.py             cửa sổ chính: ghép widget, nối với engine
│       └── widgets/           mỗi khối giao diện một file nhỏ
│           ├── header.py        tiêu đề, trạng thái mô hình, FPS
│           ├── video_panel.py   khung video
│           ├── control_bar.py   Bắt đầu/Dừng, Mở tệp, Chụp ảnh, chọn camera
│           ├── total_card.py    thẻ TỔNG TIỀN
│           ├── note_list.py     danh sách từng tờ + ô sửa tay mệnh giá
│           ├── settings_panel.py  chọn mô hình, ngưỡng tin cậy, đọc to, TTA
│           ├── action_bar.py    Ghi nhận lượt, Làm lại, Xuất CSV
│           └── status_bar.py    thời gian xử lý, thông báo ngắn
│
├── training/                HUẤN LUYỆN (TV2): prepare_data → train → export_onnx → evaluate. App không import gói này
│   ├── config.py              10 lớp + kích thước ảnh + chuẩn hoá (nguồn thứ tự lớp lúc train)
│   ├── prepare_data.py · dataset.py · train.py · export_onnx.py · evaluate.py · classifier.py
│   └── (chạy: python -m training.train ...  — xem docs/TRAINING.md)
├── models/                  MỖI MÔ HÌNH = <tên>.onnx + <tên>.classes.json (thứ tự lớp). Ứng dụng tự liệt kê, chọn ở ô "Mô hình"
│                            hiện có: currency_classifier_v1.onnx (TV Nghĩa); best_mobilenetv3.pth, confusion_matrix.png
├── notebooks/               train_mobilenetv3.ipynb (Colab), test_models.ipynb (kiểm thử cục bộ)
├── tools/check_env.py       kiểm tra môi trường
├── docs/                    MODEL_REPORT.md (báo cáo mô hình), TRAINING.md (quy trình huấn luyện TV2), CORE.md (mô tả lõi), PROJECT_CONTEXT.md
└── outputs/                 ảnh chụp, file CSV xuất ra (không đẩy lên Git)
```

## Luồng dữ liệu

```
Camera ─► NoteDetector ─► rectify ─► CurrencyClassifier ─► Aggregator ─► FrameState
 (core)     (core)        (core)          (core)             (core)      tổng tiền, các tờ
                      └──────── Pipeline ────────┘                            │
                                                                              ▼
              Engine (luồng nền) ──► Output(frame, state) ──► UI: video + thẻ tổng + danh sách
                                  └─► StableEvent ─► Speaker đọc to một lần
```

Giao diện **chỉ** nói chuyện với `Engine` (`start_camera`, `latest`, `set_manual`, `reset`...). Nhờ vậy sửa giao diện không làm hỏng lõi và ngược lại.

## Muốn sửa gì thì mở file nào?

| Việc cần làm | File |
|---|---|
| Đổi màu, phông, bo góc, kích thước cửa sổ | `app/ui/theme.py` |
| Đổi câu chữ, nhãn nút, thông báo | `app/ui/texts.py` |
| Thêm/bớt/đổi vị trí một khối trên màn hình | `app/ui/app.py` → `_build_layout()` |
| Đổi hình dạng một khối (thẻ tổng, danh sách...) | file tương ứng trong `app/ui/widgets/` |
| Đổi cách vẽ khung/nhãn trên video | `app/ui/overlay.py` |
| Chỉnh ngưỡng tin cậy, số khung bỏ phiếu, độ khắt khe cảnh báo | `app/config.py` |
| Chỉnh camera (độ phân giải, khoá nét, phơi sáng) | `app/config.py` (nhóm Camera) |
| Cải thiện việc tìm tờ tiền | `app/core/note_detector.py` (3 tầng: tìm thô → tinh chỉnh cạnh ở độ phân giải gốc → theo dõi theo thời gian; tham số ở `ClassicParams` và `DETECT_*` trong `app/config.py`). Đo trước/sau: `python tools/benchmark_detector.py --compare <note_detector_cũ.py>` (dữ liệu tổng hợp, vẫn cần thử ảnh thật). Tờ cầm tay / vật nền nhận nhầm: xem `DETECT_EXTRA`, `HIDE_NOT_MONEY*`, `FAST_START_CLASSIFY` trong `app/config.py`; soi từng ứng viên bằng `python tools/debug_detect.py anh.jpg` |
| Thêm / đổi mô hình | chép `<tên>.onnx` + `<tên>.classes.json` vào `models/` (hoặc chạy `python -m training.export_onnx`), rồi chọn ở ô **Mô hình** trong giao diện. Mô hình mở mặc định: `CLASSIFIER_PATH` trong `app/config.py` |
| Đổi cách phân loại | `app/core/classifier.py` |
| Huấn luyện / xuất mô hình mới | `training/` (xem `docs/TRAINING.md`) |
| Đổi logic tính tổng, chống nhảy số | `app/core/aggregator.py` |
| Đổi câu đọc to | `app/services/speech.py` → `say_total()` |
| Đổi định dạng file CSV | `app/services/storage.py` |

## ⚠ Những điều dễ làm hỏng

1. **Thứ tự lớp là của TỪNG mô hình và bắt buộc đúng**. Mô hình cũ (`currency_classifier_v1`, notebook Colab) train theo thứ tự chuỗi `1000, 10000, 100000, 2000, ...`; mô hình của `training/` (TV2) train theo thứ tự số `1000, 2000, 5000, 10000, ...`. Hai thứ tự **khác nhau** nên mỗi mô hình phải có file `<tên>.classes.json` riêng (`export_onnx` tự ghi). Thiếu file này, giao diện hiện cảnh báo vàng dưới ô chọn mô hình vì mệnh giá có thể bị nhận nhầm mà không báo lỗi.
2. Ảnh đưa vào mô hình phải là **RGB 224×224, chuẩn hoá ImageNet** (đã làm sẵn trong `classifier.py`).
3. Nhãn trên video chỉ dùng chữ không dấu vì `cv2.putText` không vẽ được tiếng Việt; chữ có dấu nằm ở danh sách bên phải.
4. Một số ngưỡng trong `app/config.py` đang được nới rất rộng (`MIN_AREA_RATIO = 0`, `MAX_OCCLUSION = 10`, `EDGE_MARGIN_PX = 0` nghĩa là gần như tắt kiểm tra chất lượng). Nếu muốn app khắt khe hơn thì chỉnh lại các giá trị này bằng ảnh thật.
5. Đổi ngưỡng/hằng số trong `config.py` cần **chạy lại app** mới có hiệu lực (riêng ngưỡng tin cậy và TTA chỉnh được ngay trên giao diện).

## Gợi ý chia việc

- **Giao diện:** `app/ui/` (theme, texts, widgets) — không cần hiểu OpenCV.
- **Độ chính xác nhận diện:** `app/core/note_detector.py`, `app/config.py`, `notebooks/` — thử bằng ảnh thật.
- **Tính năng thêm:** `app/services/` (đọc to, lưu trữ, thống kê).
