"""Tìm tờ tiền + 4 góc bằng OpenCV cổ điển (không cần mô hình huấn luyện).

Hợp đồng:
    đầu vào : khung hình BGR (uint8, HxWx3) từ camera
    đầu ra  : list[NoteDetection], mỗi tờ gồm
              - corners: 4 góc (4,2) theo chiều kim đồng hồ, cạnh TL->TR là cạnh DÀI,
                         cạnh dài hướng sang phải => ảnh nắn phẳng luôn nằm ngang, xoay tối đa 90°
                         (có thể ngược 180° so với chiều chữ; bộ phân loại có lượt xoay 180° bù lại)
              - corner_conf: độ tin cậy từng góc (1.0 = bình thường, thấp = nghi bị tay che / sát mép)
              - score, hand_contact, box

Ba tầng xử lý (đo trước/sau bằng tools/benchmark_detector.py):

  1. TÌM THÔ (toàn khung, ở độ rộng work_width ~480 px). Sinh nhiều giả thuyết: contour từ ảnh cạnh Canny và
     từ nhiều ngưỡng nhị phân của từng kênh, lấy bao lồi, xấp xỉ thành tứ giác. Chấm điểm: góc gần vuông,
     tỉ lệ cạnh giống tờ tiền, diện tích hợp lý, có cạnh ảnh thật dọc theo 4 cạnh. Loại trùng / lồng nhau.
     Tầng này chỉ cần đúng "đại khái" (sai vài chục pixel ở khung gốc) vì có tầng 2 sửa lại.

  2. TINH CHỈNH GÓC (ở độ phân giải GỐC). Với mỗi cạnh: lấy hàng chục điểm dọc cạnh, ở mỗi điểm quét theo
     pháp tuyến để tìm đường biên mạnh nhất NGOÀI CÙNG (nên không bị kẹt vào khung viền in bên trong tờ),
     khớp đường thẳng bền vững (Huber + loại điểm lạc), rồi lấy giao điểm hai cạnh kề làm góc mới.
     Cạnh bị che (ngón tay) thì bỏ qua cạnh đó và dùng đường cũ; cần >= 3/4 cạnh tốt mới chấp nhận.

  3. THEO DÕI THEO THỜI GIAN. Tờ của khung trước được tinh chỉnh thẳng lên khung mới (không cần tìm lại từ
     đầu), nên ít rung, ít rớt tờ và nhanh hơn; tìm thô toàn khung chỉ chạy mỗi `full_every` khung
     (hoặc ngay khi có tờ bị mất) để phát hiện tờ mới.

Giới hạn: đây là phương pháp heuristic. Nền nhiều chi tiết, ngón tay che mép, tờ gập/cong,
tờ màu gần màu nền sẽ làm giảm độ chính xác. Tờ tìm sai (vật hình chữ nhật khác) được lọc ở bước
phân loại nhờ lớp khong_phai_tien.

Thử trên ảnh của bạn (lưu ảnh có vẽ kết quả):
    python -m app.core.note_detector anh_cam_tay.jpg
Đo độ chính xác trên cảnh tổng hợp có đáp án:
    python tools/benchmark_detector.py
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
    contain_min_ratio: float = 0.50  # ... NHƯNG chỉ khi nó chiếm hơn tỉ lệ này của tứ giác lớn (tờ nhỏ đặt trên hộp/màn hình lớn thì giữ)
    contain_ratio: float = 0.80      # tứ giác nằm trong tứ giác lớn hơn quá tỉ lệ này thì bỏ
    max_candidates: int = 60         # giữ tối đa số giả thuyết điểm cao nhất để xét trùng lặp
    corner_skin_conf: float = 0.30   # độ tin cậy gán cho góc nghi bị ngón tay che
    corner_edge_conf: float = 0.50   # ... cho góc sát mép khung
    # --- tờ CẦM TRÊN TAY: bỏ vùng da khi tìm, phạt vật phẳng trơn / bàn tay ---
    skin_suppress: bool = True       # thêm một lượt tìm với vùng da bị "xoá" (ngón tay che mép không làm vỡ đường viền tờ)
    skin_suppress_min: float = 0.03  # chỉ chạy lượt đó khi da chiếm ít nhất tỉ lệ này của khung
    skin_reject: float = 0.55        # tứ giác có quá tỉ lệ này là da VÀ trơn -> là bàn tay/cánh tay, bỏ
    texture_flat: float = 0.35       # độ "trơn" dưới mức này (0-1) coi là mặt phẳng không hoa văn (hộp, màn hình, tay)
    texture_weight: float = 0.15     # trọng số điểm hoa văn: tiền in dày chi tiết, vật nền thường trơn
    texture_seeds: bool = True       # thêm ứng viên từ vùng NHIỀU HOA VĂN (tiền in dày chi tiết; ngón tay/hộp/màn hình trơn bị loại)
    texture_thresholds: tuple = (0.18, 0.28, 0.40)
    area_best: float = 0.30          # tờ chiếm tới tỉ lệ này vẫn được điểm tối đa; lớn hơn thì giảm dần (vật nền thường rất lớn)

    # --- tầng 2: tinh chỉnh góc ở độ phân giải gốc ---
    refine: bool = True
    refine_out_ratio: float = 0.13   # tầm quét ra NGOÀI = tỉ lệ này x cạnh ngắn (px khung gốc), kẹp trong [min, max]
    refine_out_min: float = 10.0
    refine_out_max: float = 30.0
    refine_in: float = 8.0           # tầm quét vào TRONG (px khung gốc)
    refine_edge_min: float = 6.0     # độ mạnh cạnh tối thiểu (độ chênh màu hai bên điểm quét, thang 0-255)
    refine_rel: float = 0.55         # chỉ nhận đỉnh cạnh >= tỉ lệ này của đỉnh mạnh nhất trên cùng điểm quét
    refine_angle_tol: float = 0.13   # độ lệch hướng cạnh cho phép so với cạnh cũ (sin ~ 7.5 độ)
    refine_min_good: int = 3         # số cạnh tốt tối thiểu (trên 4) để chấp nhận kết quả

    # --- tầng 3: theo dõi theo thời gian ---
    temporal: bool = True
    fast_px: float = 5.0             # tờ di chuyển nhanh hơn mức này (px/khung): bỏ theo dõi, tìm lại toàn khung mỗi khung
    full_every: int = 3              # tìm thô toàn khung mỗi N khung khi đang theo dõi (1 = mọi khung)
    track_radius_min: float = 6.0     # tầm quét khi theo dõi = min + track_radius_gain x tốc độ (px/khung), tối đa max
    track_radius_gain: float = 0.7
    track_radius_max: float = 26.0
    track_min_quality: float = 0.55  # chất lượng tinh chỉnh tối thiểu để giữ theo dõi
    smooth_alpha: float = 1.0        # <1: làm mượt góc giữa các khung (1 = tắt); chỉ áp khi tờ gần như đứng yên
    smooth_max_px: float = 6.0       # tờ dịch chuyển hơn mức này (px/khung) thì không làm mượt (tránh trễ)


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


def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / max(union, 1e-6))


def _intersect(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray):
    """Giao điểm hai đường thẳng (điểm + hướng đơn vị); None nếu gần song song."""
    cr = float(v1[0] * v2[1] - v1[1] * v2[0])
    if abs(cr) < 0.5:          # hai cạnh kề của tờ tiền phải gần vuông góc
        return None
    w = p2 - p1
    a = float(w[0] * v2[1] - w[1] * v2[0]) / cr
    return p1 + a * v1


@dataclass
class _Track:
    quad: np.ndarray        # (4,2) float32, toạ độ khung gốc
    score: float
    quality: float          # 0..1 chất lượng tinh chỉnh góc (0 = chưa tinh chỉnh được)
    vel: np.ndarray | None = None   # (4,2) độ dịch chuyển mỗi khung của từng góc (để dự đoán vị trí khung sau)


@dataclass(frozen=True)
class _Refined:
    quad: np.ndarray
    quality: float
    good_edges: int


# ----------------------------------------------------------------------------
class ClassicNoteDetector:
    def __init__(self, params: ClassicParams | None = None):
        self.p = params or ClassicParams()
        self.last_ms = 0.0
        self.last_info: dict = {}
        self._k3 = np.ones((3, 3), np.uint8)
        self._k5 = np.ones((5, 5), np.uint8)
        self.reset()

    def reset(self) -> None:
        """Quên các tờ đang theo dõi (gọi khi đổi nguồn ảnh / bắt đầu lượt mới)."""
        self._prev: list[_Track] = []
        self._shape: tuple[int, int] | None = None
        self._since_full = 0

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> list[NoteDetection]:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("Khung hình phải là mảng uint8 dạng HxWx3 (BGR)")
        t0 = time.perf_counter()
        p = self.p
        h0, w0 = frame_bgr.shape[:2]
        if self._shape != (h0, w0):
            self.reset()
            self._shape = (h0, w0)
        s = min(1.0, p.work_width / w0)
        w, h = (int(round(w0 * s)), int(round(h0 * s))) if s < 1 else (w0, h0)

        # --- tầng 3: tinh chỉnh thẳng các tờ của khung trước lên khung này ---
        tracked: list[_Track] = []
        lost = False
        if p.temporal and p.refine and self._prev:
            for tr in self._prev:
                vel = tr.vel if tr.vel is not None else np.zeros((4, 2), np.float32)
                pred = (tr.quad + vel).astype(np.float32)          # dự đoán: giữ nguyên vận tốc khung trước
                speed = float(np.linalg.norm(vel, axis=1).max())
                if speed > p.fast_px:                              # chuyển động nhanh: dự đoán/viền hẹp kém tin cậy
                    lost = True
                    continue
                r = self._refine(frame_bgr, pred, *self._radii(tr.quad, True, speed), tracking=True)
                if r is None or r.quality < p.track_min_quality:
                    lost = True
                    continue
                quad = self._smooth(r.quad, pred)
                tracked.append(_Track(quad, 0.85 * tr.score + 0.15 * min(1.0, 0.55 + 0.45 * r.quality), r.quality,
                                      0.5 * vel + 0.5 * (quad - tr.quad)))
        self._since_full += 1
        run_full = (not (p.temporal and p.refine) or not self._prev or lost or self._since_full >= max(1, p.full_every))
        work = None
        if run_full:                                   # --- tầng 1 (+2): tìm thô toàn khung ---
            self._since_full = 0
            work = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA) if s < 1 else frame_bgr
            tracked = self._merge(tracked, self._global_search(frame_bgr, work, s, tracked), self._prev)
        tracks = self._dedupe(tracked)[: p.max_notes]
        self._prev = tracks

        if work is None:                               # khung chỉ theo dõi: ảnh nhỏ rẻ tiền, chỉ để dò vùng da
            work = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else frame_bgr
        skin = self._skin_mask(work)
        dets = [self._make_detection(t.quad, t.score, skin, w, h, s) for t in tracks]
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.last_info = {"mode": "full" if run_full else "track", "n_tracked": len(tracks)}
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

    def _candidates(self, planes, w: int, h: int, n_thr: int | None = None):
        p = self.p
        min_a, max_a = p.min_area_ratio * w * h, p.max_area_ratio * w * h
        binaries = []
        for pl in planes:
            binaries.append(cv2.dilate(cv2.Canny(pl, 20, 60), self._k3))  # đóng khe hở trên đường biên
            for t in np.linspace(40, 215, n_thr or p.n_thresholds):
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

    def _texture_candidates(self, tex: np.ndarray, w: int, h: int):
        """Ứng viên từ các mảng hoa văn dày: tờ bị ngón tay che mép vẫn ra một mảng liền, tay/nền trơn thì không."""
        p = self.p
        min_a, max_a = p.min_area_ratio * w * h, p.max_area_ratio * w * h
        seen = set()
        for thr in p.texture_thresholds:
            m = (tex > thr).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k5)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k5)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def _score(self, quad, kind, edges, w: int, h: int, skin=None, tex=None):
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
        if ar > p.area_best:              # vật rất lớn (màn hình, bàn, sách) ít khi là tờ đang cầm
            area_score *= max(0.0, 1.0 - (ar - p.area_best) / (p.max_area_ratio - p.area_best + 1e-6))
        skin_frac, texture = self._inside_stats(q, skin, tex)
        if skin_frac > p.skin_reject and texture < p.texture_flat:
            return None                    # mảng da trơn: bàn tay / cánh tay chứ không phải tờ tiền
        score = 0.45 * support + 0.25 * fit + 0.15 * float(np.clip(asp, 0, 1)) + 0.15 * area_score
        score += p.texture_weight * (texture - 0.5)   # hoa văn dày: cộng; mặt phẳng trơn: trừ
        if kind == "rect":
            score -= 0.05
        return (q, score) if score >= p.min_score else None

    @staticmethod
    def _inside_stats(q: np.ndarray, skin, tex) -> tuple[float, float]:
        """(tỉ lệ da, độ hoa văn 0-1) bên trong tứ giác, đo trong phần lõi (co vào) để bỏ ảnh hưởng của viền."""
        if skin is None or tex is None:
            return 0.0, 0.5
        h, w = skin.shape
        x0, y0 = np.maximum(np.floor(q.min(axis=0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(q.max(axis=0)).astype(int) + 2, [w, h])
        if x1 - x0 < 6 or y1 - y0 < 6:
            return 0.0, 0.5
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cen = q.mean(axis=0)
        cv2.fillConvexPoly(m, np.round((q - cen) * 0.8 + cen - [x0, y0]).astype(np.int32), 1)   # lõi = 80% quanh tâm
        n = int(m.sum())
        if n < 20:
            return 0.0, 0.5
        sel = m.astype(bool)
        return float(skin[y0:y1, x0:x1][sel].mean()), float(tex[y0:y1, x0:x1][sel].mean())

    def _texture_map(self, plane: np.ndarray) -> np.ndarray:
        """Bản đồ độ hoa văn 0-1: biên độ gradient làm mượt (vùng phẳng ~0, vùng in nhiều chi tiết -> cao)."""
        g = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        g = np.abs(g) + np.abs(cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3))
        g = cv2.blur(g, (9, 9))
        return np.clip(g / 60.0, 0.0, 1.0)

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
                if inter / max(area + a2 - inter, 1.0) > p.nms_iou or (
                        inter / max(area, 1.0) > p.contain_ratio and area > p.contain_min_ratio * a2):
                    dup = True
                    break
            if not dup:
                accepted.append((q, sc, m, area))
        # bỏ tứ giác nằm gọn trong tứ giác lớn hơn (hình in bên trong tờ)
        keep = []
        for i, (q, sc, m, area) in enumerate(accepted):
            inside = any(
                j != i and a2 > area and area > p.contain_min_ratio * a2
                and np.count_nonzero(m & m2) / max(area, 1.0) > p.contain_ratio
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

    def _make_detection(self, quad_full, score, skin, w, h, scale) -> NoteDetection:
        """quad_full: góc ở khung gốc; scale: tỉ lệ khung gốc -> khung làm việc (để dò vùng da)."""
        p = self.p
        corners = order_corners(quad_full).astype(np.float32)
        q = corners * scale
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
        box = (float(corners[:, 0].min()), float(corners[:, 1].min()),
               float(corners[:, 0].max()), float(corners[:, 1].max()))
        return NoteDetection(box, corners, conf, float(min(1.0, score)), contact)

    # ==================================================================
    # Tầng 1: tìm thô toàn khung
    # ==================================================================
    def _global_search(self, frame_bgr: np.ndarray, work: np.ndarray, s: float,
                       tracked: list[_Track] | None = None) -> list[_Track]:
        p = self.p
        h, w = work.shape[:2]
        planes = self._planes(work)
        skin = self._skin_mask(work)
        tex = self._texture_map(planes[0])
        scored = []

        def collect(pls, n_thr=None):
            edges = self._edge_map(pls)
            for quad, kind in self._candidates(pls, w, h, n_thr):
                res = self._score(quad, kind, edges, w, h, skin, tex)
                if res is not None:
                    scored.append(res)

        collect(planes)
        if p.texture_seeds:
            edges0 = self._edge_map(planes)
            for quad, kind in self._texture_candidates(tex, w, h):
                res = self._score(quad, kind, edges0, w, h, skin, tex)
                if res is not None:
                    scored.append(res)
        if p.skin_suppress and skin.mean() >= p.skin_suppress_min:
            # lượt 2: xoá vùng da (gán bằng giá trị nền trung vị) để ngón tay/bàn tay không nối liền tờ với nền
            grown = cv2.dilate(skin.astype(np.uint8), self._k5) > 0
            sup = []
            for pl in planes:
                q = pl.copy()
                q[grown] = np.median(pl[~grown]) if (~grown).any() else 128
                sup.append(q)
            collect(sup, max(3, p.n_thresholds // 2))   # lượt phụ: ít ngưỡng hơn cho nhanh
        out: list[_Track] = []
        for q, sc in self._select(scored, w, h):
            qf = (q / s).astype(np.float32)
            if tracked and any(_poly_iou(qf, t.quad) > 0.5 for t in tracked):
                continue                                # tờ này đang được theo dõi và đã tinh chỉnh rồi
            if self.p.refine:
                r = self._refine(frame_bgr, qf, *self._radii(qf, tracking=False))
                if r is not None:
                    out.append(_Track(r.quad, sc, r.quality))
                    continue
            out.append(_Track(qf, sc, 0.0))
        return out

    # ==================================================================
    # Tầng 2: tinh chỉnh góc ở độ phân giải gốc
    # ==================================================================
    def _radii(self, quad: np.ndarray, tracking: bool, speed: float = 0.0) -> tuple[float, float]:
        """(tầm quét vào trong, tầm quét ra ngoài) tính bằng pixel khung gốc.

        Theo dõi: cửa sổ HẸP quanh vị trí dự đoán, rộng dần theo tốc độ — đủ bắt kịp tờ đang di chuyển nhưng
        không với tới khung viền in bên trong tờ hay các cạnh nền ở xa.
        """
        p = self.p
        if tracking:
            r = float(np.clip(p.track_radius_min + p.track_radius_gain * speed, p.track_radius_min, p.track_radius_max))
            return r, r
        short = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
        return p.refine_in, float(np.clip(p.refine_out_ratio * short, p.refine_out_min, p.refine_out_max))

    def _refine(self, frame: np.ndarray, quad: np.ndarray, r_in: float, r_out: float,
                tracking: bool = False) -> _Refined | None:
        """Kéo 4 cạnh của tứ giác về đúng đường biên ảnh thật; None nếu không đủ bằng chứng."""
        H, W = frame.shape[:2]
        pad = int(1.6 * max(r_in, r_out) + 8)
        x0 = max(0, int(np.floor(quad[:, 0].min())) - pad)
        x1 = min(W, int(np.ceil(quad[:, 0].max())) + pad + 1)
        y0 = max(0, int(np.floor(quad[:, 1].min())) - pad)
        y1 = min(H, int(np.ceil(quad[:, 1].max())) + pad + 1)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        roi = cv2.GaussianBlur(frame[y0:y1, x0:x1], (0, 0), 1.0).astype(np.float32)   # làm mịn nhiễu cảm biến
        origin = np.array([x0, y0], np.float64)
        q = np.asarray(quad, np.float32)
        best: _Refined | None = None
        ri, ro = r_in, r_out
        for it in range(2):                             # lần 2 quét hẹp hơn để chốt vị trí
            # Trong cửa sổ quét luôn lấy biên NGOÀI CÙNG đủ mạnh (không dính khung viền in bên trong tờ).
            res = self._refine_pass(roi, origin, q, ri, ro, prefer_outer=(it == 0 and not tracking), tight=tracking)
            if res is None:
                break
            q, quality, good = res
            best = _Refined(q, quality, good)
            ri = ro = max(3.0, 0.3 * max(ri, ro))
        return best

    def _refine_pass(self, roi, origin, q, r_in, r_out, prefer_outer: bool = True, tight: bool = False):
        p = self.p
        cen = q.mean(axis=0)
        offs = np.arange(-r_in - 1.0, r_out + 2.0, 1.0)             # dư 1 px mỗi đầu để lấy hiệu trung tâm
        edges = []                                                   # mỗi cạnh: (A, u, n, L, K, P) hoặc None
        for i in range(4):
            A = q[i].astype(np.float64)
            d = q[(i + 1) % 4].astype(np.float64) - A
            L = float(np.linalg.norm(d))
            if L < 20:
                edges.append(None)
                continue
            u = d / L
            n = np.array([u[1], -u[0]])
            if float(np.dot(n, A + d / 2.0 - cen)) < 0:             # pháp tuyến hướng ra ngoài tờ
                n = -n
            K = int(np.clip(L / 6.0, 14, 56))
            P = A[None, :] + np.linspace(0.08, 0.92, K)[:, None] * d[None, :]
            edges.append((A, u, n, L, K, P))
        live = [e for e in edges if e is not None]
        if len(live) < p.refine_min_good:
            return None

        # Lấy mẫu màu dọc pháp tuyến cho cả 4 cạnh trong MỘT lần remap
        P = np.vstack([e[5] for e in live])
        N = np.vstack([np.repeat(e[2][None, :], e[4], axis=0) for e in live])
        X = (P[:, 0:1] + offs[None, :] * N[:, 0:1] - origin[0]).astype(np.float32)
        Y = (P[:, 1:2] + offs[None, :] * N[:, 1:2] - origin[1]).astype(np.float32)
        prof = cv2.remap(roi, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)   # (R, S, 3)
        diff = prof[:, 2:, :] - prof[:, :-2, :]
        e_all = np.sqrt((diff * diff).sum(axis=2))                  # độ mạnh cạnh tại offs[1:-1]
        offs_c = offs[1:-1]
        thr = np.maximum(p.refine_edge_min, p.refine_rel * e_all.max(axis=1))
        peak = np.zeros(e_all.shape, bool)
        peak[:, 1:-1] = ((e_all[:, 1:-1] >= e_all[:, :-2]) & (e_all[:, 1:-1] > e_all[:, 2:])
                         & (e_all[:, 1:-1] >= thr[:, None]))

        lines: list = [None] * 4
        row = 0
        for i, e in enumerate(edges):
            if e is None:
                continue
            _, u, n, L, K, Pk = e
            sl = slice(row, row + K)
            row += K
            lines[i] = self._fit_profile(Pk, n, u, L, K, e_all[sl], peak[sl], offs_c, prefer_outer)
        good = [ln for ln in lines if ln is not None]
        if len(good) < p.refine_min_good:
            return None

        def line_of(i: int):                            # cạnh không khớp được -> dùng lại đường cũ
            if lines[i] is not None:
                return lines[i][0], lines[i][1]
            d = q[(i + 1) % 4] - q[i]
            return q[i].astype(np.float64), d / (np.linalg.norm(d) + 1e-9)

        out = np.empty((4, 2), np.float32)
        for i in range(4):
            pa, va = line_of((i - 1) % 4)
            pb, vb = line_of(i)
            c = _intersect(pa, va, pb, vb)
            if c is None:
                return None
            out[i] = c
        if np.linalg.norm(out - q, axis=1).max() > 1.8 * max(r_in, r_out) + 4:
            return None
        if not cv2.isContourConvex(out.reshape(-1, 1, 2)):
            return None
        a0, a1 = cv2.contourArea(q), cv2.contourArea(out)
        lo, hi = (0.85, 1.18) if tight else (0.6, 1.8)     # theo dõi: diện tích không thể đổi nhiều giữa hai khung
        if a0 <= 0 or not (lo <= a1 / a0 <= hi):
            return None
        quality = 0.5 * len(good) / 4.0 + 0.5 * float(np.mean([ln[2] for ln in good]))
        return out, quality, len(good)

    def _fit_profile(self, Pk, n, u, L, K, e, peak, offs_c, prefer_outer: bool = True):
        """Từ các đỉnh cạnh dọc một cạnh -> đường thẳng. Trả (điểm, hướng đơn vị, tỉ lệ đồng thuận) hoặc None."""
        p = self.p
        valid = peak.any(axis=1)
        nv = int(valid.sum())
        if nv < max(6, int(0.35 * K)):
            return None
        if prefer_outer:
            idx = peak.shape[1] - 1 - np.argmax(peak[:, ::-1], axis=1)     # đỉnh NGOÀI CÙNG đủ mạnh
        else:
            idx = np.argmin(np.where(peak, np.abs(offs_c)[None, :], np.inf), axis=1)   # đỉnh GẦN vị trí hiện tại nhất
        rows = np.nonzero(valid)[0]
        cols = idx[rows]
        y0, y1, y2 = e[rows, cols - 1], e[rows, cols], e[rows, cols + 1]
        den = y0 - 2 * y1 + y2
        sub = np.where(den < -1e-6, 0.5 * (y0 - y2) / np.where(den < -1e-6, den, -1.0), 0.0)
        pos = offs_c[cols] + np.clip(sub, -0.5, 0.5)
        Q = Pk[rows] + pos[:, None] * n[None, :]

        tol = 1.5 + 0.004 * L
        line = cv2.fitLine(Q.astype(np.float32).reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        v, p0 = line[:2].astype(np.float64), line[2:].astype(np.float64)
        keep = np.abs((Q - p0) @ np.array([-v[1], v[0]])) <= tol
        if keep.sum() < max(6, int(0.5 * nv)):
            return None
        line = cv2.fitLine(Q[keep].astype(np.float32).reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        v, p0 = line[:2].astype(np.float64), line[2:].astype(np.float64)
        if abs(float(u[0] * v[1] - u[1] * v[0])) > p.refine_angle_tol:   # hướng cạnh lệch quá nhiều -> đường lạ
            return None
        return p0, v, float(keep.sum()) / K

    # ==================================================================
    # Tầng 3: theo dõi
    # ==================================================================
    def _smooth(self, new: np.ndarray, old: np.ndarray) -> np.ndarray:
        a = self.p.smooth_alpha
        if a >= 1.0 or float(np.linalg.norm(new - old, axis=1).mean()) > self.p.smooth_max_px:
            return new
        return (old + a * (new - old)).astype(np.float32)

    def _merge(self, tracked: list[_Track], fresh: list[_Track], prev: list[_Track]) -> list[_Track]:
        merged = list(tracked)
        for f in fresh:
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(merged):
                iou = _poly_iou(f.quad, t.quad)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_iou > 0.3:
                t = merged[best_i]
                if best_iou < 0.85 and f.quality > t.quality + 0.05:   # hai bên bất đồng -> tin bên có bằng chứng cạnh tốt hơn
                    merged[best_i] = _Track(f.quad, f.score, f.quality, t.vel)
            else:
                f.vel = self._velocity_from(f.quad, prev)             # tờ mất dấu rồi tìm lại: kế thừa chuyển động
                merged.append(f)
        return merged

    @staticmethod
    def _velocity_from(quad: np.ndarray, prev: list[_Track]) -> np.ndarray | None:
        best, best_iou = None, 0.15
        for t in prev:
            iou = _poly_iou(quad, t.quad)
            if iou > best_iou:
                best, best_iou = t, iou
        if best is None:
            return None
        shifts = [np.roll(quad, k, axis=0) for k in range(4)]      # căn thứ tự góc theo tờ cũ
        aligned = min(shifts, key=lambda q: float(np.linalg.norm(q - best.quad, axis=1).sum()))
        return (aligned - best.quad).astype(np.float32)

    def _dedupe(self, tracks: list[_Track]) -> list[_Track]:
        out: list[_Track] = []
        for t in sorted(tracks, key=lambda x: (-x.quality, -x.score)):
            if all(_poly_iou(t.quad, o.quad) <= 0.5 for o in out):
                out.append(t)
        return out


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