# src/storage/export_excel.py
import csv
import os
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from src.storage.sqlite_db import StorageDB
from src.storage.totals import VALID_DENOMINATIONS


def export_session_to_excel(db: StorageDB, session_id: int, output_path: str) -> str:
    """Xuất phiên kiểm đếm ra file Excel (.xlsx) gồm 2 sheet: Tổng hợp và Chi tiết."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    summary = db.get_session_summary(session_id)
    details = db.get_session_details(session_id)

    wb = Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    bold_font = Font(bold=True)

    # --- SHEET 1: TỔNG HỢP ---
    ws_sum = wb.active
    ws_sum.title = "Tổng hợp"
    ws_sum.append(["Mệnh giá (VND)", "Số tờ", "Thành tiền (VND)"])

    for cell in ws_sum[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for denom in VALID_DENOMINATIONS:
        count = summary.counts_by_denom.get(denom, 0)
        amount = summary.amount_by_denom.get(denom, 0)
        ws_sum.append([denom, count, amount])

    ws_sum.append(["TỔNG CỘNG", summary.valid_note_count, summary.total_amount])
    for cell in ws_sum[ws_sum.max_row]:
        cell.font = bold_font

    # Định dạng số có dấu phẩy
    for row in ws_sum.iter_rows(min_row=2, max_row=ws_sum.max_row):
        row[0].number_format = "#,##0"
        row[2].number_format = "#,##0"

    ws_sum.column_dimensions["A"].width = 20
    ws_sum.column_dimensions["B"].width = 12
    ws_sum.column_dimensions["C"].width = 22

    # --- SHEET 2: CHI TIẾT ---
    ws_det = wb.create_sheet(title="Chi tiết")
    ws_det.append(["Lượt (Batch ID)", "Thời điểm", "Mệnh giá", "Độ tin cậy", "Sửa tay", "Ảnh lưu vết"])

    for cell in ws_det[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for item in details:
        ws_det.append([
            item["batch_id"],
            item["batch_time"],
            item["denomination"],
            round(float(item["confidence"]), 4),
            "Có" if item["is_manual_edit"] else "Không",
            item["image_path"],
        ])

    ws_det.column_dimensions["A"].width = 16
    ws_det.column_dimensions["B"].width = 22
    ws_det.column_dimensions["C"].width = 16
    ws_det.column_dimensions["D"].width = 14
    ws_det.column_dimensions["E"].width = 12
    ws_det.column_dimensions["F"].width = 35

    wb.save(output_path)
    return output_path


def export_session_to_csv(db: StorageDB, session_id: int, output_path: str) -> str:
    """Xuất danh sách chi tiết từng tờ tiền của phiên ra file CSV (UTF-8 BOM cho Excel)."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    details = db.get_session_details(session_id)
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_id", "batch_time", "denomination", "confidence", "is_manual_edit", "image_path"])
        for item in details:
            writer.writerow([
                item["batch_id"],
                item["batch_time"],
                item["denomination"],
                round(float(item["confidence"]), 4),
                "Có" if item["is_manual_edit"] else "Không",
                item["image_path"],
            ])
    return output_path