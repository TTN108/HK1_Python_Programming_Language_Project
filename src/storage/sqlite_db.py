# src/storage/sqlite_db.py
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple
from src.storage.totals import (
    BatchSummary,
    NoteRecord,
    VALID_DENOMINATIONS,
    calculate_batch_totals,
)


class StorageDB:
    """Quản lý cơ sở dữ liệu SQLite cho các phiên kiểm đếm tiền VND."""

    def __init__(self, db_path: str = "data/money_counter.db"):
        self.db_path = db_path
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    note TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    image_path TEXT DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id INTEGER NOT NULL,
                    denomination TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_manual_edit INTEGER NOT NULL DEFAULT 0,
                    bbox_json TEXT DEFAULT '[]',
                    FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE CASCADE
                );
                """
            )

    def create_session(self, note: str = "") -> int:
        """Tạo phiên kiểm quỹ mới và trả về session_id."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (started_at, note) VALUES (?, ?)",
                (now, note),
            )
            return int(cur.lastrowid)

    def save_batch(
        self,
        session_id: int,
        notes: List[NoteRecord],
        image_path: str = "",
    ) -> Tuple[int, BatchSummary]:
        """Lưu 1 lượt kiểm đếm vào phiên hiện tại và trả về (batch_id, thống kê lượt)."""
        now = datetime.now().isoformat(timespec="seconds")
        summary = calculate_batch_totals(notes)

        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO batches (session_id, created_at, image_path) VALUES (?, ?, ?)",
                (session_id, now, image_path),
            )
            batch_id = int(cur.lastrowid)

            rows = [
                (
                    batch_id,
                    str(n.denomination),
                    float(n.confidence),
                    1 if n.is_manual_edit else 0,
                    json.dumps(n.bbox or []),
                )
                for n in notes
            ]
            conn.executemany(
                """
                INSERT INTO notes (batch_id, denomination, confidence, is_manual_edit, bbox_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

        return batch_id, summary

    def get_session_details(self, session_id: int) -> List[Dict]:
        """Lấy danh sách chi tiết từng tờ tiền trong cả phiên."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    b.id AS batch_id,
                    b.created_at AS batch_time,
                    b.image_path AS image_path,
                    n.denomination AS denomination,
                    n.confidence AS confidence,
                    n.is_manual_edit AS is_manual_edit,
                    n.bbox_json AS bbox_json
                FROM notes n
                JOIN batches b ON n.batch_id = b.id
                WHERE b.session_id = ?
                ORDER BY b.id ASC, n.id ASC
                """,
                (session_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_session_summary(self, session_id: int) -> BatchSummary:
        """Tính tổng hợp cộng dồn của tất cả các lượt trong một phiên."""
        details = self.get_session_details(session_id)
        notes = [
            NoteRecord(
                denomination=row["denomination"],
                confidence=row["confidence"],
                is_manual_edit=bool(row["is_manual_edit"]),
                bbox=json.loads(row["bbox_json"]) if row["bbox_json"] else [],
            )
            for row in details
        ]
        return calculate_batch_totals(notes)

    def list_sessions(self) -> List[Dict]:
        """Liệt kê tất cả các phiên đã lưu."""
        with self._connect() as conn:
            cur = conn.execute("SELECT id, started_at, note FROM sessions ORDER BY id DESC")
            return [dict(row) for row in cur.fetchall()]