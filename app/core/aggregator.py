"""Theo dõi từng tờ qua các khung hình, bỏ phiếu mệnh giá, chống nhảy số, hỗ trợ sửa tay.

Hợp đồng:
    update(list[Detection]) -> FrameState
    FrameState.tracks: list[TrackView] (trái -> phải) với status:
        'stable'    mệnh giá đã chốt bằng bỏ phiếu, được cộng tiền
        'manual'    người dùng sửa tay (một mệnh giá), được cộng tiền
        'not_money' lớp khong_phai_tien (bỏ phiếu hoặc sửa tay), không cộng
        'uncertain' chưa chốt được (cần chờ hoặc sửa tay), không cộng
        'warning'   đang có cảnh báo chồng / sát mép / bị che, không cộng
    FrameState.total: tổng tiền các tờ 'stable' và 'manual'.
Tờ mất tạm thời <= MAX_MISSED_FRAMES khung vẫn được giữ nên tổng không nhảy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from app import config
from app.core.pipeline import Detection

UNCERTAIN_KEY = "?"
STATUS_STABLE = "stable"
STATUS_MANUAL = "manual"
STATUS_NOT_MONEY = "not_money"
STATUS_UNCERTAIN = "uncertain"
STATUS_WARNING = "warning"
COUNTED = (STATUS_STABLE, STATUS_MANUAL)


@dataclass
class _Track:
    id: int
    box: tuple[float, float, float, float]
    corners: np.ndarray
    reasons: tuple[str, ...] = ()
    history: deque = field(default_factory=lambda: deque(maxlen=config.HISTORY_LEN))
    missed: int = 0
    age: int = 1
    manual: str | None = None
    guess: str | None = None        # nhãn dự đoán của khung gần nhất (để hiển thị gợi ý)
    guess_conf: float = 0.0


@dataclass(frozen=True)
class TrackView:
    id: int
    box: tuple[float, float, float, float]
    corners: np.ndarray = field(repr=False, compare=False)
    label: str | None          # nhãn đã chốt hoặc sửa tay; None nếu chưa chốt
    value: int                 # số tiền được cộng (0 nếu chưa/không cộng)
    status: str
    confidence: float          # tỉ lệ đồng thuận trong cửa sổ bỏ phiếu (0..1)
    manual: bool
    reasons: tuple[str, ...]
    visible: bool              # False = đang mất tạm thời
    age: int
    guess: str | None
    guess_conf: float


@dataclass(frozen=True)
class FrameState:
    tracks: list[TrackView]
    total: int
    n_money: int               # số tờ được cộng tiền
    n_unrecognized: int        # số tờ chưa chốt (uncertain)
    n_warning: int
    n_not_money: int
    stable: bool               # tổng đã giữ nguyên đủ TOTAL_STABLE_FRAMES khung và không có cảnh báo
    just_stabilized: bool      # đúng khung vừa chuyển sang ổn định (để phát âm báo một lần)
    stable_frames: int
    released_frames: int       # số khung liên tiếp không thấy tờ nào (khay trống)

    @property
    def has_warning(self) -> bool:
        return self.n_warning > 0

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)


def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


class Aggregator:
    def __init__(self):
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._empty_frames = 0
        self._stable_frames = 0
        self._last_sig: tuple | None = None
        self.state: FrameState = self._build_state(False)

    # ------------------------------------------------------------------ API
    def update(self, detections: Sequence[Detection]) -> FrameState:
        matches = self._match(detections)
        matched_t = {ti for ti, _ in matches}
        matched_d = {di for _, di in matches}

        for ti, di in matches:
            t, d = self._tracks[ti], detections[di]
            t.missed = 0
            t.age += 1
            self._apply(t, d)
        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= config.MAX_MISSED_FRAMES]
        for di, d in enumerate(detections):
            if di not in matched_d:
                t = _Track(self._next_id, d.box, d.corners)
                self._next_id += 1
                self._apply(t, d)
                self._tracks.append(t)

        visible = any(t.missed == 0 for t in self._tracks)
        self._empty_frames = 0 if visible else self._empty_frames + 1
        self.state = self._build_state(True)
        return self.state

    def set_manual(self, track_id: int, label: str | None) -> bool:
        """Sửa tay nhãn một tờ (label=None để trả về tự động). False nếu không có tờ đó."""
        if label is not None and label not in config.CLASSES:
            raise ValueError(f"Lớp không hợp lệ: {label!r}")
        for t in self._tracks:
            if t.id == track_id:
                t.manual = label
                self.state = self._build_state(True)
                return True
        return False

    def reset(self) -> None:
        """Xoá toàn bộ track (làm lại lượt)."""
        self._tracks = []
        self._empty_frames = 0
        self._stable_frames = 0
        self._last_sig = None
        self.state = self._build_state(False)

    # ------------------------------------------------------------ nội bộ
    def _apply(self, t: _Track, d: Detection) -> None:
        t.box, t.corners, t.reasons = d.box, d.corners, d.reasons
        res = d.result
        if res is not None:
            t.guess, t.guess_conf = res.label, res.confidence
        if res is None or any(r in config.NO_VOTE_REASONS for r in d.reasons):
            return
        key = res.label if res.status in ("ok", "not_money") else UNCERTAIN_KEY
        t.history.append((key, max(config.VOTE_WEIGHT_MIN, d.rect.quality)))

    def _match(self, dets: Sequence[Detection]) -> list[tuple[int, int]]:
        pairs = []
        for ti, t in enumerate(self._tracks):
            long_side = max(t.box[2] - t.box[0], t.box[3] - t.box[1], 1.0)
            tc = np.array([(t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2])
            for di, d in enumerate(dets):
                iou = _poly_iou(t.corners, d.corners)
                if iou >= config.TRACK_MIN_IOU:
                    pairs.append((1.0 + iou, ti, di))
                    continue
                dc = np.array([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
                dist = float(np.linalg.norm(tc - dc)) / long_side
                if dist <= config.TRACK_MAX_CENTER_DIST:
                    pairs.append((1.0 - dist / config.TRACK_MAX_CENTER_DIST, ti, di))
        pairs.sort(key=lambda p: -p[0])
        used_t, used_d, out = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            out.append((ti, di))
        return out

    @staticmethod
    def _decide(t: _Track) -> tuple[str | None, float]:
        if len(t.history) < config.MIN_VOTES:
            return None, 0.0
        weights: dict[str, float] = {}
        for key, w in t.history:
            weights[key] = weights.get(key, 0.0) + w
        total = sum(weights.values())
        key, w = max(weights.items(), key=lambda kv: kv[1])
        share = w / total if total > 0 else 0.0
        if key == UNCERTAIN_KEY or share < config.CONSENSUS_RATIO:
            return None, share
        return key, share

    def _view(self, t: _Track) -> TrackView:
        warnings = [r for r in t.reasons if r in config.WARNING_REASONS]
        if t.manual:  # người dùng đã nhìn tận mắt nên bỏ qua cảnh báo "bị che" (heuristic)
            warnings = [r for r in warnings if r != "occluded"]
        label, share = self._decide(t)
        if t.manual:
            label, share = t.manual, 1.0
        if warnings:
            status, value = STATUS_WARNING, 0
        elif label is None:
            status, value = STATUS_UNCERTAIN, 0
        elif label == config.NOT_MONEY_CLASS:
            status, value = STATUS_NOT_MONEY, 0
        else:
            status = STATUS_MANUAL if t.manual else STATUS_STABLE
            value = config.class_to_value(label)
        return TrackView(t.id, t.box, t.corners, label, value, status, share, bool(t.manual),
                         t.reasons, t.missed == 0, t.age, t.guess, t.guess_conf)

    def _build_state(self, advance: bool) -> FrameState:
        views = sorted((self._view(t) for t in self._tracks), key=lambda v: v.box[0])
        total = sum(v.value for v in views if v.status in COUNTED)
        n_money = sum(v.status in COUNTED for v in views)
        n_unrec = sum(v.status == STATUS_UNCERTAIN for v in views)
        n_warn = sum(v.status == STATUS_WARNING for v in views)
        n_not = sum(v.status == STATUS_NOT_MONEY for v in views)

        just = False
        if advance:
            ok = n_money > 0 and n_warn == 0 and n_unrec == 0
            sig = (total, n_money)
            if ok:
                self._stable_frames = self._stable_frames + 1 if sig == self._last_sig else 1
                just = self._stable_frames == config.TOTAL_STABLE_FRAMES
            else:
                self._stable_frames = 0
            self._last_sig = sig
        stable = self._stable_frames >= config.TOTAL_STABLE_FRAMES
        return FrameState(views, total, n_money, n_unrec, n_warn, n_not, stable, just,
                          self._stable_frames, self._empty_frames)