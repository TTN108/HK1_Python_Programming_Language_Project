# 🧠 BỐI CẢNH & NHẬT KÝ ĐỒ ÁN THỊ GIÁC MÁY TÍNH
## Đề tài: Nhận diện Mệnh giá Tiền Việt Nam (VN Banknote Recognition)

---

### 1. Thông tin chung & Mục tiêu Đề tài
* **Tên đề tài:** Ứng dụng Thị giác máy tính nhận diện và phân loại mệnh giá tiền Việt Nam theo thời gian thực (Realtime).
* **Mục tiêu chính:**
  * Sử dụng webcam/camera máy tính để quét và phát hiện các tờ tiền polymer Việt Nam.
  * Khoanh vùng (Bounding Box) và phân loại chính xác các mệnh giá: **10.000đ, 20.000đ, 50.000đ, 100.000đ, 200.000đ, 500.000đ**.
  * Đếm số lượng tờ tiền xuất hiện trong khung hình và tính tổng số tiền (Tổng giá trị).
  * **Tính năng nhân văn & thực tiễn cao:** Tích hợp bộ phát giọng nói tiếng Việt offline (`pyttsx3`) đọc to mệnh giá và tổng tiền nhằm hỗ trợ người khiếm thị.

---

### 2. Cấu hình Môi trường Kỹ thuật (Technical Specifications)
* **Hệ điều hành:** Windows (x64).
* **Trình soạn thảo mã (IDE):** Visual Studio Code.
* **Quản lý phiên bản mã nguồn:** Git (2.56+) kết hợp **GitHub Desktop** (giao diện đồ họa trực quan cho làm việc nhóm, không cần cài GitLens).
* **Python chuẩn hệ thống:** **Python 3.12.10** (Bản cài đặt chính thức của Windows).
  * ⚠️ *Lưu ý quan trọng:* Trên máy có môi trường MSYS2 (dùng cho C++) chứa Python 3.14.5. **Tuyệt đối không sử dụng Python 3.14.5** vì chưa có các gói thư viện AI tương thích (OpenCV, PyTorch sẽ lỗi 100%).
* **Môi trường ảo (Virtual Environment):** Đã khởi tạo độc lập tại thư mục `D:\do_an_python\.venv`.
  * Trình thông dịch: `D:\do_an_python\.venv\Scripts\python.exe`.
  * Cấu hình tự động nhận diện trong `.vscode/settings.json`.

---

### 3. Danh mục Thư viện đã Cài đặt trong `.venv` & Vai trò Chi tiết
Tất cả các thư viện sau đã được cài đặt và kiểm thử thành công:

| Thư viện | Phiên bản | Vai trò trong Đồ án |
| :--- | :--- | :--- |
| **`opencv-python`** | 5.0.0.93 | Mở webcam, bắt khung hình thời gian thực (realtime), tiền xử lý ảnh, vẽ khung nhận diện. |
| **`ultralytics`** | 8.4.198 | Huấn luyện và thực thi mô hình Deep Learning **YOLO (YOLOv8 / YOLO11)** - mô hình nhận diện vật thể nhanh và chính xác nhất hiện nay. |
| **`torch` & `torchvision`** | 2.15.0 / 0.29.2 | Framework Deep Learning cốt lõi chạy ngầm bên dưới YOLO. |
| **`pillow` (PIL)** | 13.0.0 | **Rất quan trọng:** Hàm vẽ chữ của OpenCV (`cv2.putText`) không hỗ trợ tiếng Việt có dấu. Dùng Pillow để vẽ nhãn tiếng Việt chuẩn UTF-8 (ví dụ: *"500.000 VNĐ - Năm trăm nghìn đồng"*). |
| **`pyttsx3`** | 2.99 | Chuyển văn bản thành giọng nói (Text-To-Speech) hoàn toàn **offline**, phát âm thanh thông báo mệnh giá qua loa máy tính. |
| **`numpy`** | 2.5.4 | Xử lý ma trận dữ liệu ảnh đa chiều, tính toán ma trận nhanh. |
| **`matplotlib`** | 3.11.2 | Vẽ biểu đồ huấn luyện (mAP, Precision, Recall, Loss curve) để đưa vào báo cáo và slide đồ án. |
| **`ipykernel`** | 7.4.0 | Cầu nối Kernel giúp chạy file notebook `.ipynb` trực tiếp trên VS Code bằng môi trường `.venv`. |

---

### 4. Chiến lược Phối hợp Phần cứng & Đám mây (Hardware & Cloud Workflow)
Do laptop cá nhân sử dụng GPU Onboard / chưa cấu hình NVIDIA CUDA:
1. **Làm việc trên máy cá nhân (VS Code):**
   * Viết code xử lý luồng, giao diện hiển thị, kết nối webcam realtime (`cv2.VideoCapture(0)`).
   * Viết module xử lý âm thanh tiếng Việt (`pyttsx3`).
   * Kiểm thử ứng dụng hoàn chỉnh với file trọng số mô hình đã huấn luyện.
2. **Huấn luyện mô hình trên Đám mây (Google Colab / Kaggle):**
   * Tập dữ liệu ảnh (Dataset) sau khi gán nhãn sẽ được đưa lên **Google Colab**.
   * Tận dụng **GPU T4 miễn phí** của Google để train YOLOv8/YOLO11 trong 15–30 phút.
   * Sau khi train xong, tải file trọng số tốt nhất `best.pt` về máy cá nhân, đặt vào dự án tại `D:\do_an_python\models\best.pt` để ứng dụng trên VS Code load lên chạy.

---

### 5. Kinh nghiệm Xương máu cho Đề tài Nhận diện Tiền Việt Nam
1. **Tránh nhầm lẫn màu sắc:**
   * Tờ **20.000đ** và **500.000đ** cùng tông màu xanh lơ/xanh lam.
   * Tờ **10.000đ** và **200.000đ** cùng tông màu hơi nâu/đỏ cam.
   * *Giải pháp:* Khi chụp ảnh dữ liệu, phải chụp cả mặt trước (ảnh Bác Hồ), mặt sau (danh lam thắng cảnh) với nhiều góc chụp, khoảng cách, độ gấp nếp và điều kiện ánh sáng (sáng rõ, ngược sáng, thiếu sáng).
2. **Quy trình gán nhãn dữ liệu (Data Labeling):**
   * Sử dụng nền tảng **Roboflow (miễn phí)** để cả nhóm cùng tải ảnh lên và kéo khung gán nhãn (bounding boxes).
   * Xuất dataset theo định dạng chuẩn: **YOLOv8 PyTorch format**.

---

### 6. Lộ trình Triển khai Đồ án (Roadmap)
- [x] **Giai đoạn 1: Thiết lập & Kiểm thử Môi trường (ĐÃ HOÀN TẤT 100%)**
  - Khởi tạo `.venv` với Python 3.12.10.
  - Cài đặt đầy đủ bộ thư viện CV & AI.
  - Smoke test qua script `test_doan.py` (in thông số & phát âm thanh kiểm tra loa thành công).
- [ ] **Giai đoạn 2: Thu thập & Chuẩn bị Dữ liệu (Dataset Collection)**
  - Chụp ảnh 6 mệnh giá tiền giấy/polymer (mỗi mệnh giá tối thiểu 100–200 ảnh đa dạng góc độ).
- [ ] **Giai đoạn 3: Gán nhãn dữ liệu (Annotation / Labeling)**
  - Tải lên Roboflow, phân chia Train/Valid/Test set (tỉ lệ 70:20:10).
  - Tăng cường dữ liệu (Data Augmentation: xoay, đổi sáng, làm mờ nhẹ).
- [ ] **Giai đoạn 4: Huấn luyện Mô hình trên Google Colab**
  - Viết notebook train YOLOv8n / YOLO11n với GPU T4.
  - Đánh giá chỉ số mAP50, mAP50-95.
  - Tải file weights `best.pt`.
- [ ] **Giai đoạn 5: Xây dựng Ứng dụng Hoàn chỉnh trên VS Code**
  - Viết module nhận diện webcam thời gian thực (`detect_webcam.py`).
  - Tích hợp tiếng Việt có dấu qua Pillow.
  - Bộ đếm tiền & tính tổng tiền.
  - Luồng phát âm thanh tiếng Việt (`pyttsx3`) không gây giật lag hình ảnh (sử dụng Threading/Queue).
- [ ] **Giai đoạn 6: Hoàn thiện Báo cáo, Đóng gói & Thuyết trình**
