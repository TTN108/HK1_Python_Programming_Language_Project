"""Ghép ba bước: tìm tờ (note_detector) -> nắn phẳng (rectify) -> phân loại (classifier).

Hợp đồng:
    process(frame_bgr) -> (list[Detection], timings)
    Detection: note (NoteDetection), rect (RectifiedNote),
               result (ClassifyResult | None: None nếu tờ quá nhỏ hoặc khung này bỏ qua phân loại theo classify_every),
               reasons (lý do chất lượng, gồm cả 'overlap' khi hai tờ giao nhau)
    timings: {'detect_ms', 'rectify_ms', 'classify_ms'} để hiện trên thanh trạng thái
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

import config
from classifier import ClassifyResult
from note_detector import ClassicNoteDetector, ClassicParams, NoteDetection
from rectify import RectifiedNote, rectify


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


class Pipeline:
    def __init__(self, classifier, detector=None, max_notes: int = config.MAX_NOTES,
                 classify_every: int = config.CLASSIFY_EVERY):
        self.classifier = classifier
        self.detector = detector or ClassicNoteDetector(
            ClassicParams(max_notes=max_notes, work_width=config.DETECT_WORK_WIDTH))
        self.classify_every = max(1, int(classify_every))
        self._frame_no = 0

    def process(self, frame_bgr: np.ndarray) -> tuple[list[Detection], dict]:
        t0 = time.perf_counter()
        notes = self.detector.detect(frame_bgr)
        t1 = time.perf_counter()
        rects = [rectify(frame_bgr, n) for n in notes]
        t2 = time.perf_counter()

        # Khung bỏ qua phân loại thì result=None: aggregator vẫn cập nhật vị trí nhưng không bỏ phiếu.
        classify_now = self._frame_no % self.classify_every == 0
        self._frame_no += 1
        to_classify = ([i for i, r in enumerate(rects) if "too_small" not in r.reasons]
                       if classify_now else [])
        results: list[ClassifyResult | None] = [None] * len(notes)
        if to_classify:
            out = self.classifier.predict_batch([rects[i].image for i in to_classify])
            for i, res in zip(to_classify, out):
                results[i] = res
        t3 = time.perf_counter()

        overlap = _overlapping([n.corners for n in notes], config.OVERLAP_RATIO)
        dets = []
        for i, (n, r, res) in enumerate(zip(notes, rects, results)):
            reasons = r.reasons + (("overlap",) if i in overlap else ())
            dets.append(Detection(n, r, res, reasons))
        timings = {
            "detect_ms": (t1 - t0) * 1000.0,
            "rectify_ms": (t2 - t1) * 1000.0,
            "classify_ms": (t3 - t2) * 1000.0,
        }
        return dets, timings