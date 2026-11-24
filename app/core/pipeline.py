"""Ghép ba bước: tìm tờ (note_detector) -> nắn phẳng (rectify) -> phân loại (classifier).

Hợp đồng:
    process(frame_bgr) -> (list[Detection], timings)
    Detection: note (NoteDetection), rect (RectifiedNote),
               result (ClassifyResult | None: None nếu tờ quá nhỏ hoặc khung này bỏ qua phân loại theo classify_every),
               reasons (lý do chất lượng, gồm cả 'overlap' khi hai tờ giao nhau)
    timings: {'detect_ms', 'rectify_ms', 'classify_ms', 'hidden'} để hiện trên thanh trạng thái
             ('hidden' = số vật bị ẩn vì bộ phân loại xác nhận không phải tiền)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from app import config
from app.core.classifier import ClassifyResult
from app.core.note_detector import ClassicNoteDetector, ClassicParams, NoteDetection
from app.core.rectify import RectifiedNote, rectify


@dataclass(frozen=True)
class Detection:
    note: NoteDetection
    rect: RectifiedNote
    result: ClassifyResult | None          # None nếu tờ quá nhỏ nên không phân loại
    reasons: tuple[str, ...]

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.note.box

    @property
    def corners(self) -> np.ndarray:
        return self.note.corners


def _overlapping(corners: list[np.ndarray], ratio: float) -> set[int]:
    """Chỉ số các tờ giao nhau với tờ khác quá `ratio` (so với tờ nhỏ hơn)."""
    flagged: set[int] = set()
    areas = [cv2.contourArea(c.astype(np.float32)) for c in corners]
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            inter, _ = cv2.intersectConvexConvex(corners[i].astype(np.float32), corners[j].astype(np.float32))
            if inter > ratio * max(min(areas[i], areas[j]), 1.0):
                flagged.update((i, j))
    return flagged


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class _Verdict:
    """Trí nhớ ngắn về một vật mà bộ tìm đưa ra: bộ phân loại đã nói gì về nó qua các lần phân loại."""
    quad: np.ndarray
    n: int = 0              # số lần đã phân loại
    nm: int = 0             # số lần phân loại liền nhau cho kết quả "không phải tiền" (đủ chắc)
    last: int = 0           # số thứ tự khung lần cuối thấy

    @property
    def rejected(self) -> bool:
        return self.nm >= config.NOT_MONEY_CONFIRM


class Pipeline:
    """tìm tờ -> nắn phẳng -> phân loại, kèm LỌC bằng bộ phân loại.

    Bộ tìm chỉ dựa vào hình học nên đưa ra cả hộp, màn hình, sách phía sau. Bộ phân loại (học từ ảnh tiền thật) biết
    đâu là tiền: vật nào được xác nhận là "khong_phai_tien" vài lần liền sẽ bị ẩn (không vào bộ đếm, không vẽ lên video).
    Vật mới xuất hiện được phân loại NGAY (không chờ tới lượt) để tờ tiền thật hiện kết quả nhanh.
    """

    def __init__(self, classifier, detector=None, max_notes: int = config.MAX_NOTES,
                 classify_every: int = config.CLASSIFY_EVERY):
        self.classifier = classifier
        self.max_notes = max_notes
        self.detector = detector or ClassicNoteDetector(
            ClassicParams(max_notes=max_notes + config.DETECT_EXTRA, work_width=config.DETECT_WORK_WIDTH,
                          refine=config.DETECT_REFINE, temporal=config.DETECT_TEMPORAL,
                          full_every=config.DETECT_FULL_EVERY))
        self.classify_every = max(1, int(classify_every))
        self.hide_not_money = config.HIDE_NOT_MONEY
        self._frame_no = 0
        self._verdicts: list[_Verdict] = []
        self.debug: list[dict] = []        # ứng viên khung gần nhất (kể cả bị ẩn) — cho tools/debug_detect.py

    def reset(self) -> None:
        self._verdicts = []
        self._frame_no = 0
        if hasattr(self.detector, "reset"):
            self.detector.reset()

    def _verdict_for(self, corners: np.ndarray) -> _Verdict:
        best, best_iou = None, 0.5
        for v in self._verdicts:
            iou = _quad_iou(corners, v.quad)
            if iou > best_iou:
                best, best_iou = v, iou
        if best is None:
            best = _Verdict(corners.copy())
            self._verdicts.append(best)
        best.quad, best.last = corners.copy(), self._frame_no
        return best

    def process(self, frame_bgr: np.ndarray) -> tuple[list[Detection], dict]:
        t0 = time.perf_counter()
        notes = self.detector.detect(frame_bgr)
        t1 = time.perf_counter()
        rects = [rectify(frame_bgr, n) for n in notes]
        t2 = time.perf_counter()

        regular = self._frame_no % self.classify_every == 0
        self._frame_no += 1
        verdicts = [self._verdict_for(n.corners) for n in notes]
        self._verdicts = [v for v in self._verdicts if self._frame_no - v.last <= 15]

        # Phân loại: tới lượt, hoặc vật còn mới (chưa đủ vài lần phân loại -> cần kết quả nhanh).
        # Vật đã bị ẩn chỉ kiểm tra lại thỉnh thoảng cho đỡ tốn thời gian.
        to_classify = []
        for i, (r, v) in enumerate(zip(rects, verdicts)):
            if "too_small" in r.reasons:
                continue
            fresh = v.n < config.FAST_START_CLASSIFY
            if v.rejected:
                due = regular and (self._frame_no // self.classify_every) % config.REJECT_RECHECK == 0
            else:
                due = regular or fresh
            if due:
                to_classify.append(i)
        results: list[ClassifyResult | None] = [None] * len(notes)
        if to_classify:
            out = self.classifier.predict_batch([rects[i].image for i in to_classify])
            for i, res in zip(to_classify, out):
                results[i] = res
                v = verdicts[i]
                v.n += 1
                sure_not_money = res.status == "not_money" and res.confidence >= config.HIDE_NOT_MONEY_CONF
                v.nm = v.nm + 1 if sure_not_money else 0
        t3 = time.perf_counter()

        keep = [i for i in range(len(notes))
                if not (self.hide_not_money and verdicts[i].rejected)][: self.max_notes]
        hidden = len(notes) - len(keep)
        self.debug = [dict(corners=notes[i].corners, score=notes[i].score,
                           label=results[i].label if results[i] else None,
                           conf=results[i].confidence if results[i] else None,
                           hidden=i not in keep) for i in range(len(notes))]
        overlap = _overlapping([notes[i].corners for i in keep], config.OVERLAP_RATIO)
        dets = []
        for k, i in enumerate(keep):
            reasons = rects[i].reasons + (("overlap",) if k in overlap else ())
            dets.append(Detection(notes[i], rects[i], results[i], reasons))
        timings = {
            "detect_ms": (t1 - t0) * 1000.0,
            "rectify_ms": (t2 - t1) * 1000.0,
            "classify_ms": (t3 - t2) * 1000.0,
            "hidden": hidden,
        }
        return dets, timings
