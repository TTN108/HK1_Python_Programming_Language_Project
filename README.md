# Module Lưu trữ, Tính tổng & Kiểm thử (TV4 – `src/storage/` & `tests/`)

Module chịu trách nhiệm tính tổng tiền theo mệnh giá, lưu lịch sử kiểm đếm offline bằng SQLite, xuất báo cáo Excel/CSV và cung cấp bộ kiểm thử tự động bằng `pytest`.

---

## 1. Luồng xử lý (Pipeline)

```text
[Danh sách tờ tiền từ TV1/TV2: (mệnh giá, độ tin cậy, sửa tay, tọa độ)]
       │
       ├──► (totals.py)      : Tính tổng, đếm số tờ 9 mệnh giá, kiểm tra tờ "không chắc chắn"
       │
       ├──► (sqlite_db.py)   : Lưu vào 3 bảng SQLite (sessions -> batches -> notes)
       │
       └──► (export_excel.py): Xuất file .xlsx (sheet Tổng hợp & Chi tiết) hoặc .csv
```

### Hợp đồng giao tiếp với TV1

- **Đầu vào:** Danh sách đối tượng `NoteRecord(denomination, confidence, is_manual_edit, bbox)`.
- **Đầu ra:** `BatchSummary` gồm `total_amount`, `counts_by_denom`, `amount_by_denom`, `uncertain_count`, `is_final` và `display_label` (`"Tổng"` hoặc `"Tổng tạm tính (còn N tờ chưa xác nhận)"`).

---

## 2. Yêu cầu hệ thống

- **Ngôn ngữ:** Python 3.11.
- **Thư viện:** `openpyxl` (xuất Excel), `pytest` (kiểm thử tự động). `sqlite3` và `csv` dùng trực tiếp từ thư viện chuẩn Python.

---

## 3. Mô tả từng module

| File | Vai trò chính |
| :--- | :--- |
| `src/storage/totals.py` | Định nghĩa `NoteRecord`, `BatchSummary` và hàm `calculate_batch_totals()` để cộng mệnh giá hợp lệ, bỏ qua vật lạ và đếm tờ khung vàng. |
| `src/storage/sqlite_db.py` | Lớp `StorageDB` khởi tạo 3 bảng `sessions`, `batches`, `notes`; hỗ trợ tạo phiên, lưu lượt, cộng dồn cả phiên và truy vấn chi tiết. |
| `src/storage/export_excel.py` | Xuất báo cáo Excel 2 sheet (`Tổng hợp` và `Chi tiết`, có đánh dấu tờ sửa tay và đường dẫn ảnh lưu vết) và xuất file CSV. |
| `tests/test_storage.py` | Bộ unit test tự động cho hàm tính tổng, lưu trữ SQLite và xuất file Excel/CSV. |

---

## 4. Thử nhanh bằng lệnh (PowerShell / Terminal)

> Chạy các lệnh dưới đây từ **thư mục gốc của dự án**.

### Bước 1: Cài đặt thư viện

```powershell
pip install -r requirements-tv4.txt
```

### Bước 2: Chạy bộ kiểm thử tự động `pytest`

```powershell
python -m pytest -v tests/test_storage.py
```
