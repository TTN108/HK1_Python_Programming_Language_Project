> **Lưu ý:** đây là tài liệu của phần lõi xử lý do TV3 viết. Sau khi tái cấu trúc, các file `.py` nằm trong `app/core/` và cấu hình ở `app/config.py`; phần giao diện nằm ở `app/ui/`. Xem `README.md` ở thư mục gốc để biết bức tranh chung.

# Nhận diện & đếm tiền VND bằng camera

Lõi xử lý (không gồm giao diện) của ứng dụng nhận diện mệnh giá tiền Việt Nam từ camera/webcam hoặc file video. Ứng dụng tìm từng tờ tiền trong khung hình, nắn phẳng, phân loại mệnh giá bằng mô hình ONNX (MobileNetV3), theo dõi qua nhiều khung hình bằng cách bỏ phiếu và cho ra **tổng tiền ổn định, không nhảy số**.

## Luồng xử lý

```
Camera ──► NoteDetector ──► rectify ──► CurrencyClassifier ──► Aggregator ──► FrameState
(khung BGR) (tìm 4 góc)   (ảnh 224x224)  (mệnh giá + độ tin cậy) (theo dõi, bỏ phiếu)  (tổng tiền, trạng thái)
                └──────────────── Pipeline ────────────────┘
```

| Bước | File | Việc làm |
|---|---|---|
| 1. Đọc khung | `app/core/camera.py` | Đọc camera/video ở thread nền, luôn giữ khung mới nhất, đo FPS |
| 2. Tìm tờ | `app/core/note_detector.py` | Tìm tờ tiền và 4 góc bằng OpenCV cổ điển (không cần huấn luyện) |
| 3. Nắn phẳng | `app/core/rectify.py` | Nắn theo 4 góc thành ảnh 224x224 RGB, tính độ nét, mức bị che, chất lượng |
| 4. Phân loại | `app/core/classifier.py` | ONNX Runtime, trả mệnh giá + độ tin cậy, có TTA xoay 180° |
| Ghép 2–4 | `app/core/pipeline.py` | Chạy tuần tự các bước trên một khung, gắn cảnh báo chồng tờ (`overlap`) |
| 5. Tổng hợp | `app/core/aggregator.py` | Theo dõi từng tờ, bỏ phiếu, chống nhảy số, hỗ trợ sửa tay |
| Cấu hình | `app/config.py` | Toàn bộ hằng số và ngưỡng |

## Cài đặt

Yêu cầu Python 3.10+ (mã dùng cú pháp `X | None`).

```bash
pip install opencv-python numpy onnxruntime
# có GPU NVIDIA: dùng onnxruntime-gpu thay cho onnxruntime
```

Mô hình mặc định: `models/currency_classifier_v1.onnx` (`CLASSIFIER_PATH` trong `app/config.py`). Mọi `.onnx` trong `models/` đều chọn được trong giao diện (`app/core/model_registry.py`, `Engine.load_model`). Thứ tự lớp của từng mô hình nằm ở `models/<tên>.classes.json`.

### Mô hình & thứ tự lớp

- Đầu vào: NCHW, 3 kênh, RGB, chuẩn hoá `/255` rồi trừ mean/chia std ImageNet. **Phải khớp lúc train.**
- Hỗ trợ batch động hoặc batch cố định = 1; float32 hoặc float16.
- Đầu ra là xác suất hoặc logit (tự nhận biết và softmax).
- Số lớp của mô hình phải bằng số lớp trong file lớp của nó, nếu lệch sẽ báo lỗi rõ ràng.
- Thứ tự lớp **phải đúng thứ tự đầu ra lúc train**, và là của TỪNG mô hình. Ưu tiên: `models/<tên>.classes.json` (riêng) → `models/classes.json` (chung, chỉ để tương thích) → thứ tự chuỗi kiểu `ImageFolder`. Mô hình cũ train theo thứ tự chuỗi (`1000, 10000, 100000, 2000, …`), mô hình `training/` theo thứ tự số (`1000, 2000, 5000, 10000, …`). File lớp là mảng hoặc `{"classes": [...]}`, có đúng một lớp `khong_phai_tien`:

```json
{"classes": ["1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000", "500000", "khong_phai_tien"]}
```

Tên lớp đọc được các dạng `50000`, `50.000`, `50k`, `500K`.

## Dùng nhanh

### Thử từng bước bằng dòng lệnh

```bash
# Tìm tờ trên ảnh, lưu ảnh vẽ kết quả (<tên>_detect.jpg)
python -m app.core.note_detector anh_cam_tay.jpg --min-score 0.5 --max-notes 5

# Phân loại ảnh tờ đã cắt/nắn phẳng
python -m app.core.classifier to1.jpg to2.jpg --conf 0.7 --no-tta
```

### Dùng trong mã

```python
from app.core.camera import Camera
from app.core.classifier import CurrencyClassifier
from app.core.pipeline import Pipeline
from app.core.aggregator import Aggregator

clf = CurrencyClassifier()
clf.warmup()
pipe = Pipeline(clf)
agg = Aggregator()
cam = Camera(source=0).start()          # hoặc đường dẫn video

while True:
    frame, frame_id = cam.read(timeout=1.0)
    if frame is None:
        if cam.ended:
            break
        continue
    detections, timings = pipe.process(frame)
    state = agg.update(detections)

    print(state.total, state.stable)
    if state.just_stabilized:
        pass  # phát âm báo một lần tại đây

cam.stop()
```

## Mô tả từng module

### `app/config.py`
Gộp mọi hằng số: đường dẫn, camera, tốc độ xử lý, ngưỡng nắn phẳng, bộ phân loại, theo dõi, âm báo. Các ngưỡng chỉ là điểm khởi đầu, **cần chỉnh lại bằng ảnh thật**. Khi đóng gói `.exe`, thư mục `outputs/` nằm cạnh file chạy.

### `app/core/camera.py`
- `Camera(source, width, height, fps, ...)`: nguồn là số (webcam), đường dẫn video hoặc URL.
- `read(timeout)` chờ khung **mới hơn** lần đọc trước, trả `(None, id)` nếu hết giờ hoặc video đã hết.
- Webcam: ép MJPG để đạt HD 30 FPS, tuỳ chọn khoá phơi sáng/cân bằng trắng, khoá nét (`focus`), đặt thời gian phơi sáng (`exposure`). Thuộc tính driver từ chối được ghi vào `camera.warnings`.
- Windows dùng DirectShow; `show_dialog=True` mở hộp thoại thuộc tính driver để chỉnh tay rồi ghi số vào `app/config.py`.
- Video file phát theo tốc độ gốc, có thể lặp (`loop_video=True`).

### `app/core/note_detector.py`
Bộ tìm tờ cổ điển (`ClassicNoteDetector`, tham số trong `ClassicParams`):
1. Thu khung về rộng `work_width`, tách các kênh L, a, b, S.
2. Sinh giả thuyết từ contour của ảnh cạnh Canny và nhiều ngưỡng nhị phân, lấy bao lồi rồi xấp xỉ thành tứ giác (hoặc hình chữ nhật nhỏ nhất nếu bị che góc).
3. Chấm điểm: góc gần vuông, tỉ lệ cạnh giống tờ tiền, diện tích hợp lý, và có cạnh ảnh thật dọc theo cạnh.
4. Loại trùng và tứ giác nằm trong tứ giác khác, giữ tối đa `max_notes` tờ.

Mỗi `NoteDetection` gồm `corners` (4 góc theo chiều kim đồng hồ, cạnh TL→TR là cạnh dài), `corner_conf`, `score`, `hand_contact` (tỉ lệ vùng da quanh tờ) và `box`. Giao diện `NoteDetector` (Protocol) cho phép sau này thay bằng bản YOLO (`create_detector("yolo")` hiện chưa làm).

**Giới hạn**: nền nhiều chi tiết, ngón tay che mép, tờ gập/cong, tờ màu gần màu nền đều làm giảm độ chính xác. Vật hình chữ nhật khác bị lọc ở bước phân loại nhờ lớp `khong_phai_tien`.

### `app/core/rectify.py`
`rectify(frame, det)` trả `RectifiedNote`:
- `image`: 224x224, RGB, đưa thẳng vào bộ phân loại.
- `sharpness` (phương sai Laplacian), `area_ratio`, `occlusion` (0..1), `quality` (0..1, dùng làm trọng số phiếu).
- `reasons`: `blur`, `too_small`, `occluded`, `cut_by_edge`.

### `app/core/classifier.py`
`CurrencyClassifier.predict / predict_batch` trả `ClassifyResult(label, confidence, status, value, runner_up, probs)`:

| `status` | Ý nghĩa | Cộng tiền |
|---|---|---|
| `ok` | độ tin cậy ≥ `CONF_THRESHOLD` | có |
| `uncertain` | độ tin cậy thấp, cần chờ hoặc sửa tay | không |
| `not_money` | lớp `khong_phai_tien` | không |

TTA: tờ có độ tin cậy đầu < `TTA_SKIP_CONF` được chạy thêm lượt xoay 180° và lấy trung bình (bù lại việc ảnh nắn phẳng có thể ngược chiều). Tự chọn `CUDAExecutionProvider` nếu có.

### `app/core/pipeline.py`
`Pipeline.process(frame_bgr) -> (list[Detection], timings)`.
- Chỉ phân loại mỗi `classify_every` khung (mặc định 2); khung bỏ qua vẫn theo dõi vị trí nhưng `result=None` nên không bỏ phiếu.
- Tờ `too_small` không được phân loại.
- Gắn `overlap` cho các tờ giao nhau quá `OVERLAP_RATIO`.
- `timings`: `detect_ms`, `rectify_ms`, `classify_ms`.

### `app/core/aggregator.py`
`Aggregator.update(detections) -> FrameState`:
- **Ghép track** theo IoU đa giác (`TRACK_MIN_IOU`) hoặc khoảng cách tâm (`TRACK_MAX_CENTER_DIST`).
- **Bỏ phiếu** trong cửa sổ `HISTORY_LEN` khung, trọng số theo chất lượng khung; chốt nhãn khi có ≥ `MIN_VOTES` phiếu và đồng thuận ≥ `CONSENSUS_RATIO`.
- **Chống nhảy số**: tờ mất tạm thời ≤ `MAX_MISSED_FRAMES` khung vẫn được giữ.
- **Sửa tay**: `set_manual(track_id, label)` (truyền `None` để trả về tự động); sửa tay bỏ qua cảnh báo `occluded`.
- `reset()` xoá toàn bộ track để làm lượt mới.

Trạng thái mỗi tờ (`TrackView.status`):

| Trạng thái | Ý nghĩa | Cộng tiền |
|---|---|---|
| `stable` | mệnh giá đã chốt bằng bỏ phiếu | có |
| `manual` | người dùng sửa tay | có |
| `not_money` | không phải tiền | không |
| `uncertain` | chưa chốt được | không |
| `warning` | chồng tờ / sát mép / bị che | không |

`FrameState` gồm `tracks` (trái → phải), `total`, `n_money`, `n_unrecognized`, `n_warning`, `n_not_money`, `stable` (tổng giữ nguyên `TOTAL_STABLE_FRAMES` khung và không có cảnh báo), `just_stabilized` (đúng khung vừa ổn định, dùng để phát âm báo một lần), `released_frames` (số khung liên tiếp khay trống).

## Các tham số hay chỉnh

| Tham số (`app/config.py`) | Tác dụng |
|---|---|
| `CONF_THRESHOLD` | ngưỡng tin cậy để cộng tiền |
| `CLASSIFY_EVERY`, `DETECT_WORK_WIDTH` | đổi tốc độ lấy độ chính xác |
| `MIN_SHARPNESS`, `MAX_OCCLUSION`, `EDGE_MARGIN_PX`, `OVERLAP_RATIO` | độ khắt khe của cảnh báo chất lượng |
| `HISTORY_LEN`, `MIN_VOTES`, `CONSENSUS_RATIO` | độ nhạy/độ ổn định của bỏ phiếu |
| `MAX_MISSED_FRAMES` | thời gian giữ tờ khi bị mất |
| `TOTAL_STABLE_FRAMES` | số khung để coi tổng là ổn định |
| `FOCUS_VALUE`, `EXPOSURE_VALUE`, `LOCK_EXPOSURE` | khoá nét/phơi sáng cho webcam |


## Lọc vật nền bằng bộ phân loại & tờ cầm tay (Pipeline)

- Bộ tìm (hình học + vùng hoa văn dày, có lượt tìm thứ hai "xoá vùng da" để ngón tay không làm vỡ viền tờ) đưa ra `MAX_NOTES + DETECT_EXTRA` ứng viên.
- `Pipeline` phân loại tất cả; vật bị kết luận `khong_phai_tien` (độ tin cậy ≥ `HIDE_NOT_MONEY_CONF`) `NOT_MONEY_CONFIRM` lần liền sẽ **bị ẩn** (không đếm, không vẽ). Thanh trạng thái hiện "ẩn N vật không phải tiền"; công tắc "Ẩn vật không phải tiền" trong cài đặt để tắt khi nghi ẩn nhầm.
- Vật mới được phân loại mọi khung cho tới `FAST_START_CLASSIFY` lần, rồi mới theo `CLASSIFY_EVERY`; `MIN_VOTES`=3 và `TOTAL_STABLE_FRAMES`=12 để chốt nhanh hơn.
- Gỡ lỗi trên ảnh thật: `python tools/debug_detect.py anh.jpg` vẽ mọi ứng viên (xanh = giữ, đỏ = ẩn) kèm điểm và nhãn.
