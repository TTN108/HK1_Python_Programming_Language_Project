"""Tìm tờ tiền + 4 góc bằng OpenCV cổ điển (không cần mô hình huấn luyện).

Hợp đồng:
    đầu vào : khung hình BGR (uint8, HxWx3) từ camera
    đầu ra  : list[NoteDetection], mỗi tờ gồm
              - corners: 4 góc (4,2) theo chiều kim đồng hồ, cạnh TL->TR là cạnh DÀI,
                         cạnh dài hướng sang phải => ảnh nắn phẳng luôn nằm ngang, xoay tối đa 90°
                         (có thể ngược 180° so với chiều chữ; bộ phân loại có lượt xoay 180° bù lại)
              - corner_conf: độ tin cậy từng góc (1.0 = bình thường, thấp = nghi bị tay che / sát mép)
              - score, hand_contact, box

Cách làm (giống bộ tìm hình chữ nhật/tài liệu kinh điển):
    1. Thu khung về chiều rộng ~640 px, tách các kênh L, a, b, S.
    2. Sinh nhiều giả thuyết: contour từ ảnh cạnh Canny và từ nhiều ngưỡng nhị phân của từng kênh,
       lấy bao lồi, xấp xỉ thành tứ giác (hoặc hình chữ nhật nhỏ nhất nếu bị che góc).
    3. Chấm điểm từng tứ giác: góc gần vuông, tỉ lệ cạnh giống tờ tiền (~1.8-2.6),
       diện tích hợp lý, và quan trọng nhất là có cạnh ảnh thật dọc theo 4 cạnh.
    4. Loại trùng / tứ giác nằm trong tứ giác khác, giữ tối đa max_notes tờ.

Giới hạn: đây là phương pháp heuristic. Nền nhiều chi tiết, ngón tay che mép, tờ gập/cong,
tờ màu gần màu nền sẽ làm giảm độ chính xác. Tờ tìm sai (vật hình chữ nhật khác) được lọc ở bước
phân loại nhờ lớp khong_phai_tien.

Thử trên ảnh của bạn (lưu ảnh có vẽ kết quả):
    python note_detector.py anh_cam_tay.jpg
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Tham số (giá trị khởi đầu, chỉnh bằng ảnh thật)
# ----------------------------------------------------------------------------
@dataclass
class ClassicParams:
    work_width: int = 640            # thu khung về rộng tối đa chừng này trước khi xử lý
    max_notes: int = 5               # số tờ tối đa trả về
    min_area_ratio: float = 0.015    # tờ chiếm tối thiểu tỉ lệ này của khung
    max_area_ratio: float = 0.60     # tối đa (lớn hơn thường là phát hiện sai)
    aspect_min: float = 1.5          # tỉ lệ cạnh dài/ngắn (tờ VND ~1.85-2.4, cộng dư cho phối cảnh)
    aspect_max: float = 3.2
    max_cos: float = 0.40            # góc phải gần vuông: |cos| lớn nhất cho phép (~66°-114°)
    min_fill: float = 0.88           # (bao lồi / hình chữ nhật nhỏ nhất) tối thiểu khi dùng nhánh dự phòng
    min_edge_support: float = 0.45   # tỉ lệ điểm trên cạnh có đường biên ảnh thật (trung bình 3 cạnh tốt nhất)
    min_score: float = 0.55
    n_thresholds: int = 6            # số ngưỡng nhị phân mỗi kênh
    nms_iou: float = 0.40
    contain_ratio: float = 0.80      # tứ giác nằm trong tứ giác lớn hơn quá tỉ lệ này thì bỏ
    max_candidates: int = 60         # giữ tối đa số giả thuyết điểm cao nhất để xét trùng lặp
    corner_skin_conf: float = 0.30   # độ tin cậy gán cho góc nghi bị ngón tay che
    corner_edge_conf: float = 0.50   # ... cho góc sát mép khung


@dataclass(frozen=True)
class NoteDetection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (toạ độ khung gốc)
    corners: np.ndarray                     # (4,2) float32, theo chiều kim đồng hồ, TL->TR là cạnh dài
    corner_conf: np.ndarray                 # (4,) float32 trong 0..1
    score: float                            # 0..1, điểm tin cậy của việc đây là một tờ hình chữ nhật
    hand_contact: float                     # 0..1, tỉ lệ vùng da quanh tờ (ước lượng mức đang cầm/che)
    source: str = "classic"


class NoteDetector(Protocol):
    """Giao diện chung: bản cổ điển hôm nay, bản YOLO sau này thay vào mà phần còn lại không đổi."""

    def detect(self, frame_bgr: np.ndarray) -> list[NoteDetection]: ...


# ----------------------------------------------------------------------------
# Hình học
# ----------------------------------------------------------------------------
def order_corners(pts) -> np.ndarray:
    """Sắp 4 điểm theo chiều kim đồng hồ, p0->p1 là cạnh dài và hướng sang phải."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    # trục y của ảnh hướng xuống nên góc tăng dần = chiều kim đồng hồ trên màn hình
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    d01 = np.linalg.norm(pts[0] - pts[1]) + np.linalg.norm(pts[2] - pts[3])
    d12 = np.linalg.norm(pts[1] - pts[2]) + np.linalg.norm(pts[3] - pts[0])
    if d12 > d01:
        pts = np.roll(pts, -1, axis=0)
    v = pts[1] - pts[0]
    if v[0] < -1e-6 or (abs(v[0]) <= 1e-6 and v[1] < 0):
        pts = np.roll(pts, -2, axis=0)
    return np.ascontiguousarray(pts)


def _cos_stats(q: np.ndarray) -> tuple[float, float]:
    cs = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cs.append(abs(float(np.dot(a, b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)))
    return float(np.mean(cs)), float(np.max(cs))


def _poly_mask(q: np.ndarray, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 1)
    return m


# ----------------------------------------------------------------------------
class ClassicNoteDetector:
    def __init__(self, params: ClassicParams | None = None):
        self.p = params or ClassicParams()
        self.last_ms = 0.0
        self._k3 = np.ones((3, 3), np.uint8)
        self._k5 = np.ones((5, 5), np.uint8)

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> list[NoteDetection]:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("Khung hình phải là mảng uint8 dạng HxWx3 (BGR)")
        t0 = time.perf_counter()
        p = self.p
        h0, w0 = frame_bgr.shape[:2]
        s = min(1.0, p.work_width / w0)
        work = cv2.resize(frame_bgr, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA) if s < 1 else frame_bgr
        h, w = work.shape[:2]

        planes = self._planes(work)
        edges = self._edge_map(planes)
        scored = []
        for quad, kind in self._candidates(planes, w, h):
            res = self._score(quad, kind, edges, w, h)
            if res is not None:
                scored.append(res)
        picked = self._select(scored, w, h)

        skin = self._skin_mask(work)
        dets = [self._make_detection(q, sc, skin, w, h, 1.0 / s) for q, sc in picked]
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return dets

    # ------------------------------------------------------------------
    def _planes(self, work: np.ndarray) -> list[np.ndarray]:
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        sat = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[..., 1]
        planes = [lab[..., 0], lab[..., 1], lab[..., 2], sat]
        return [cv2.GaussianBlur(np.ascontiguousarray(pl), (5, 5), 0) for pl in planes]

    def _edge_map(self, planes) -> np.ndarray:
        e = np.zeros(planes[0].shape, np.uint8)
        for pl in planes[:3]:
            e |= cv2.Canny(pl, 25, 75)
        return cv2.dilate(e, self._k3) > 0

    def _candidates(self, planes, w: int, h: int):
        p = self.p
        min_a, max_a = p.min_area_ratio * w * h, p.max_area_ratio * w * h
        binaries = []
        for pl in planes:
            binaries.append(cv2.dilate(cv2.Canny(pl, 20, 60), self._k3))  # đóng khe hở trên đường biên
            for t in np.linspace(40, 215, p.n_thresholds):
                binaries.append(cv2.threshold(pl, int(t), 255, cv2.THRESH_BINARY)[1])
        seen = set()
        for b in binaries:
            contours, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_a or area > max_a:
                    continue
                res = self._hull_to_quad(cv2.convexHull(c))
                if res is None:
                    continue
                quad, kind = res
                key = (tuple((quad.mean(axis=0) / 4).astype(int)), int(np.sqrt(cv2.contourArea(quad)) / 4))
                if key in seen:
                    continue
                seen.add(key)
                yield quad, kind

    def _hull_to_quad(self, hull: np.ndarray):
        per = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.045, 0.06):
            ap = cv2.approxPolyDP(hull, eps * per, True)
            if len(ap) == 4:
                return ap.reshape(4, 2).astype(np.float32), "poly"
            if len(ap) < 4:
                break
        # nhánh dự phòng: góc bị che làm bao lồi không phải tứ giác -> dùng hình chữ nhật nhỏ nhất
        rect = cv2.minAreaRect(hull)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area > 0 and cv2.contourArea(hull) / rect_area >= self.p.min_fill:
            return cv2.boxPoints(rect).astype(np.float32), "rect"
        return None

    def _edge_support(self, q: np.ndarray, edges: np.ndarray) -> float:
        h, w = edges.shape
        fracs = []
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            n = max(8, int(np.linalg.norm(b - a) / 3))
            t = np.linspace(0.05, 0.95, n)[:, None]
            pts = a + (b - a) * t
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            fracs.append(float(edges[ys, xs].mean()))
        return float(np.mean(sorted(fracs)[-3:]))  # cho phép một cạnh bị che

    def _score(self, quad, kind, edges, w: int, h: int):
        p = self.p
        mean_cos, max_cos = _cos_stats(quad)
        if max_cos > p.max_cos:
            return None
        q = order_corners(quad)
        a = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3])) / 2
        b = (np.linalg.norm(q[1] - q[2]) + np.linalg.norm(q[3] - q[0])) / 2
        ratio = float(a / max(b, 1e-6))
        if not (p.aspect_min <= ratio <= p.aspect_max):
            return None
        ar = cv2.contourArea(q) / float(w * h)
        if not (p.min_area_ratio <= ar <= p.max_area_ratio):
            return None
        support = self._edge_support(q, edges)
        if support < p.min_edge_support:
            return None
        fit = float(np.clip(1.0 - mean_cos / p.max_cos, 0.0, 1.0))
        if ratio < 1.8:
            asp = (ratio - p.aspect_min) / (1.8 - p.aspect_min)
        elif ratio > 2.6:
            asp = (p.aspect_max - ratio) / (p.aspect_max - 2.6)
        else:
            asp = 1.0
        area_score = min(1.0, ar / 0.06)  # ưu tiên khung ngoài hơn các hình chữ nhật nhỏ in bên trong tờ
        score = 0.45 * support + 0.25 * fit + 0.15 * float(np.clip(asp, 0, 1)) + 0.15 * area_score
        if kind == "rect":
            score -= 0.05
        return (q, score) if score >= p.min_score else None

    def _select(self, scored, w: int, h: int):
        p = self.p
        scored = sorted(scored, key=lambda x: -x[1])[: p.max_candidates]
        accepted = []  # (quad, score, mask, area)
        for q, sc in scored:
            m = _poly_mask(q, w, h)
            area = float(m.sum())
            dup = False
            for _, _, m2, a2 in accepted:
                inter = float(np.count_nonzero(m & m2))
                if inter / max(area + a2 - inter, 1.0) > p.nms_iou or inter / max(area, 1.0) > p.contain_ratio:
                    dup = True
                    break
            if not dup:
                accepted.append((q, sc, m, area))
        # bỏ tứ giác nằm gọn trong tứ giác lớn hơn (hình in bên trong tờ)
        keep = []
        for i, (q, sc, m, area) in enumerate(accepted):
            inside = any(
                j != i and a2 > area and np.count_nonzero(m & m2) / max(area, 1.0) > p.contain_ratio
                for j, (_, _, m2, a2) in enumerate(accepted)
            )
            if not inside:
                keep.append((q, sc))
        return keep[: p.max_notes]

    # ------------------------------------------------------------------
    def _skin_mask(self, work: np.ndarray) -> np.ndarray:
        ycc = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))
        return cv2.morphologyEx(skin, cv2.MORPH_OPEN, self._k5) > 0

    def _make_detection(self, q, score, skin, w, h, up) -> NoteDetection:
        p = self.p
        conf = np.ones(4, np.float32)
        for i, (x, y) in enumerate(q):
            xi, yi = int(round(x)), int(round(y))
            win = skin[max(0, yi - 5): yi + 6, max(0, xi - 5): xi + 7]
            if win.size and win.mean() > 0.30:
                conf[i] = p.corner_skin_conf
            if xi < 3 or yi < 3 or xi > w - 4 or yi > h - 4:
                conf[i] = min(conf[i], p.corner_edge_conf)
        m = _poly_mask(q, w, h)
        ring = (cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0) & (m == 0)
        contact = float((skin & ring).sum() / max(ring.sum(), 1))
        corners = (q * up).astype(np.float32)
        box = (float(corners[:, 0].min()), float(corners[:, 1].min()),
               float(corners[:, 0].max()), float(corners[:, 1].max()))
        return NoteDetection(box, corners, conf, float(min(1.0, score)), contact)


def create_detector(kind: str = "classic", **kwargs) -> NoteDetector:
    if kind == "classic":
        return ClassicNoteDetector(**kwargs)
    if kind == "yolo":
        raise NotImplementedError("Bản YOLO chưa làm; hiện chỉ có kind='classic'")
    raise ValueError(f"kind không hợp lệ: {kind!r}")


# ----------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, dets: list[NoteDetection]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        pts = np.round(d.corners).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        for i, ((x, y), c) in enumerate(zip(pts, d.corner_conf)):
            cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0) if c >= 0.9 else (0, 0, 255), -1)
            cv2.putText(out, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"{d.score:.2f}", (int(pts[:, 0].min()), int(pts[:, 1].min()) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Thử bộ tìm tờ tiền cổ điển trên ảnh; lưu ảnh có vẽ kết quả")
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="đường dẫn ảnh kết quả (mặc định: <tên>_detect.jpg)")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--max-notes", type=int, default=None)
    args = ap.parse_args()

    params = ClassicParams()
    if args.min_score is not None:
        params.min_score = args.min_score
    if args.max_notes is not None:
        params.max_notes = args.max_notes
    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)  # đọc được đường dẫn có dấu
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {args.image}")
    det = ClassicNoteDetector(params)
    dets = det.detect(img)
    print(f"{len(dets)} tờ, {det.last_ms:.0f} ms")
    for i, d in enumerate(dets, 1):
        print(f"  #{i} điểm {d.score:.2f}, tiếp xúc tay {d.hand_contact:.0%}, độ tin cậy góc {d.corner_conf.round(1).tolist()}")
    out_path = Path(args.out) if args.out else Path(args.image).with_name(Path(args.image).stem + "_detect.jpg")
    ok, buf = cv2.imencode(".jpg", draw_detections(img, dets))
    if ok:
        buf.tofile(str(out_path))
        print(f"Đã lưu: {out_path}")


if __name__ == "__main__":
    _main()