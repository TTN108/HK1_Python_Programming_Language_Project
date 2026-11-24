"""Đọc to tổng tiền bằng tiếng Việt (offline, qua pyttsx3). Chạy ở luồng riêng nên không làm đơ giao diện.

    speaker = Speaker()
    speaker.say_total(120000, 3)     # "Ba tờ. Tổng cộng một trăm hai mươi nghìn đồng"
    speaker.enabled = False          # tắt tiếng

Không cài pyttsx3 hoặc máy không có giọng đọc thì Speaker.available = False và mọi lệnh say() bị bỏ qua,
ứng dụng vẫn chạy bình thường.
"""
from __future__ import annotations

import queue
import threading

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_UNITS = ["", "nghìn", "triệu", "tỷ"]


def _read_three(n: int, full: bool) -> str:
    """Đọc nhóm 3 chữ số (0..999). full=True: nhóm không phải nhóm đầu, phải đọc cả 'không trăm'."""
    h, rest = divmod(n, 100)
    t, u = divmod(rest, 10)
    words: list[str] = []
    if h or full:
        words.append(f"{_DIGITS[h]} trăm")
    if t > 1:
        words.append(f"{_DIGITS[t]} mươi")
        if u == 1:
            words.append("mốt")
        elif u == 5:
            words.append("lăm")
        elif u:
            words.append(_DIGITS[u])
    elif t == 1:
        words.append("mười")
        if u == 5:
            words.append("lăm")
        elif u:
            words.append(_DIGITS[u])
    elif u:
        if h or full:
            words.append("lẻ")
        words.append(_DIGITS[u])
    return " ".join(words)


def number_to_vietnamese(n: int) -> str:
    """500000 -> 'năm trăm nghìn'; 1205000 -> 'một triệu hai trăm lẻ năm nghìn'."""
    if n < 0:
        return "âm " + number_to_vietnamese(-n)
    if n == 0:
        return "không"
    if n >= 10**12:
        return str(n)
    groups: list[int] = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)
    parts: list[str] = []
    for i in reversed(range(len(groups))):
        if groups[i] == 0:
            continue
        text = _read_three(groups[i], full=i < len(groups) - 1)
        parts.append(f"{text} {_UNITS[i]}".strip())
    return " ".join(parts)


class Speaker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.available = True
        self.note = ""                       # lý do nếu không dùng được (hiện trên giao diện)
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="speaker")
        self._thread.start()

    # ------------------------------------------------------------------ API
    def say(self, text: str) -> None:
        if not (self.enabled and self.available):
            return
        while not self._q.empty():           # bỏ câu cũ chưa đọc, chỉ đọc câu mới nhất
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(text)

    def say_total(self, total: int, n_notes: int) -> None:
        self.say(f"{number_to_vietnamese(n_notes).capitalize()} tờ. "
                 f"Tổng cộng {number_to_vietnamese(total)} đồng")

    def close(self) -> None:
        self._q.put(None)

    # ------------------------------------------------------------------ luồng nền
    def _worker(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()          # phải khởi tạo ngay trong luồng này (Windows/COM)
            self._pick_vietnamese_voice(engine)
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.note = f"Không dùng được giọng đọc: {exc}"
            return
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:  # noqa: BLE001 - lỗi loa không được làm sập ứng dụng
                pass

    @staticmethod
    def _pick_vietnamese_voice(engine) -> None:
        for voice in engine.getProperty("voices"):
            blob = f"{voice.id} {voice.name} {getattr(voice, 'languages', '')}".lower()
            if "vietnam" in blob or "vi-vn" in blob or "vi_vn" in blob:
                engine.setProperty("voice", voice.id)
                return
