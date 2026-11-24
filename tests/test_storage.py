# tests/test_storage.py
from openpyxl import load_workbook
from src.storage.export_excel import export_session_to_csv, export_session_to_excel
from src.storage.sqlite_db import StorageDB
from src.storage.totals import NoteRecord, calculate_batch_totals


def test_calculate_totals_all_valid():
    notes = [
        NoteRecord("50000", 0.98),
        NoteRecord("50000", 0.95),
        NoteRecord("200000", 0.99),
        NoteRecord("10000", 0.92, is_manual_edit=True),
    ]
    res = calculate_batch_totals(notes)
    assert res.total_amount == 310000
    assert res.valid_note_count == 4
    assert res.uncertain_count == 0
    assert res.counts_by_denom[50000] == 2
    assert res.is_final is True
    assert res.display_label == "Tổng"


def test_calculate_totals_with_uncertain_and_non_money():
    notes = [
        NoteRecord("100000", 0.97),
        NoteRecord("không chắc chắn", 0.65),
        NoteRecord("khong_phai_tien", 0.94),
    ]
    res = calculate_batch_totals(notes)
    assert res.total_amount == 100000
    assert res.valid_note_count == 1
    assert res.uncertain_count == 1
    assert res.is_final is False
    assert "Tổng tạm tính" in res.display_label


def test_sqlite_save_and_export_excel_csv(tmp_path):
    db_file = tmp_path / "test_history.db"
    db = StorageDB(str(db_file))

    session_id = db.create_session(note="Kiểm quỹ cuối ngày")
    db.save_batch(
        session_id,
        [NoteRecord("20000", 0.96), NoteRecord("50000", 0.91, is_manual_edit=True)],
        image_path="sessions/s1/batch_1.jpg",
    )
    db.save_batch(
        session_id,
        [NoteRecord("500000", 0.99)],
        image_path="sessions/s1/batch_2.jpg",
    )

    summary = db.get_session_summary(session_id)
    assert summary.total_amount == 570000
    assert summary.valid_note_count == 3

    # Kiểm thử xuất Excel
    xlsx_path = tmp_path / "report.xlsx"
    export_session_to_excel(db, session_id, str(xlsx_path))
    assert xlsx_path.exists()

    wb = load_workbook(xlsx_path)
    assert wb.sheetnames == ["Tổng hợp", "Chi tiết"]
    ws_sum = wb["Tổng hợp"]
    assert ws_sum[ws_sum.max_row][2].value == 570000

    # Kiểm thử xuất CSV
    csv_path = tmp_path / "report.csv"
    export_session_to_csv(db, session_id, str(csv_path))
    assert csv_path.exists()