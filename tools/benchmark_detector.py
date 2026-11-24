"""Đo độ chính xác + tốc độ của bộ tìm tờ tiền trên cảnh TỔNG HỢP có đáp án góc.

    python tools/benchmark_detector.py                      # đo bộ tìm hiện tại (app/core/note_detector.py)
    python tools/benchmark_detector.py --seqs 40 --frames 30
    python tools/benchmark_detector.py --compare cu/note_detector.py   # so sánh với một bản khác

Mỗi "chuỗi" là một cảnh 1280x720 gồm 1-3 tờ giả di chuyển (đứng yên / chậm / nhanh), có nền đa dạng, bóng đổ,
phối cảnh, mờ chuyển động, nhiễu, ngón tay che góc, độ tương phản thấp và vật hình chữ nhật gây nhiễu.
Vì ảnh là tổng hợp nên con số chỉ để SO SÁNH TRƯỚC/SAU khi sửa thuật toán, không thay cho kiểm tra bằng ảnh thật.

Các chỉ số (toạ độ khung gốc 1280x720):
    recall    : % tờ thật được tìm thấy với IoU >= 0.7
    iou       : IoU trung bình của các tờ tìm thấy
    err_px    : sai số góc trung bình (pixel); p90 = phân vị 90
    fp/khung  : số phát hiện không khớp tờ nào (gồm cả vật gây nhiễu hình chữ nhật)
    rot_khung : % lần tờ đang được tìm thấy ở khung trước nhưng mất ở khung sau (làm số nhảy)
    rung_px   : độ lệch chuẩn vị trí góc khi tờ đứng yên (càng nhỏ càng ổn định)
    ms        : thời gian xử lý trung bình / phân vị 95 của riêng bộ tìm tờ
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

W, H = 1280, 720
PALETTE = [(60, 110, 170), (170, 130, 60), (150, 90, 190), (90, 160, 100),
           (70, 70, 190), (160, 170, 70), (110, 150, 200), (190, 170, 150)]
SKIN = (115, 150, 205)


# ----------------------------------------------------------------------------
# Sinh ảnh tổng hợp
# ----------------------------------------------------------------------------
def _smooth_noise(rng, h, w, scale, amp):
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale), 3)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def make_background(rng) -> np.ndarray:
    kind = int(rng.integers(0, 5))
    base = rng.integers(140, 240, 3).astype(np.float32)
    if kind == 1:                                   # vân gỗ
        base = np.array([70, 110, 160], np.float32) + rng.normal(0, 15, 3)
    elif kind == 2:                                 # thảm tối
        base = rng.integers(25, 80, 3).astype(np.float32)
    img = np.empty((H, W, 3), np.float32)
    img[:] = base
    img += _smooth_noise(rng, H, W, 90, 10)
    if kind == 1:
        y = np.arange(H, dtype=np.float32)[:, None, None]
        img += 14 * np.sin(y * 0.07 + _smooth_noise(rng, H, W, 60, 3)[..., :1])
    elif kind == 3:                                 # nền lộn xộn
        for _ in range(int(rng.integers(15, 35))):
            col = tuple(int(v) for v in rng.integers(0, 255, 3))
            p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            p2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            shape = int(rng.integers(0, 3))
            overlay = img.copy()
            if shape == 0:
                cv2.line(overlay, p1, p2, col, int(rng.integers(1, 6)), cv2.LINE_AA)
            elif shape == 1:
                cv2.rectangle(overlay, p1, p2, col, -1)
            else:
                cv2.circle(overlay, p1, int(rng.integers(10, 80)), col, -1, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    elif kind == 4:                                 # vignette
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        img *= (1.0 - 0.35 * np.clip(d, 0, 1.2))[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_texture(rng, long_side: int, ratio: float, low_contrast_to: np.ndarray | None) -> np.ndarray:
    w, h = long_side, int(long_side / ratio)
    c1 = np.array(PALETTE[int(rng.integers(len(PALETTE)))], np.float32) + rng.normal(0, 12, 3)
    if low_contrast_to is not None:                 # tờ có màu gần màu nền
        c1 = low_contrast_to.astype(np.float32) + rng.normal(0, 10, 3)
    c2 = c1 + rng.normal(0, 28, 3)
    gx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    ang = float(rng.uniform(0, np.pi))
    t = np.clip((gx * np.cos(ang) + gy * np.sin(ang)) / (abs(np.cos(ang)) + abs(np.sin(ang)) + 1e-6), 0, 1)
    img = np.clip(c1 * (1 - t) + c2 * t + _smooth_noise(rng, h, w, 20, 9), 0, 255).astype(np.uint8)

    def col(v):
        return tuple(int(x) for x in np.clip(v, 0, 255))

    inset = max(3, int(0.045 * h))
    cv2.rectangle(img, (inset, inset), (w - inset, h - inset), col(c1 * 0.55), 3)
    xs = np.arange(0, w, 4)
    for _ in range(int(rng.integers(6, 14))):
        y0, amp, f, ph = h * rng.uniform(0.15, 0.85), h * rng.uniform(0.02, 0.08), rng.uniform(0.01, 0.05), rng.uniform(0, 6.28)
        pts = np.stack([xs, y0 + amp * np.sin(xs * f + ph)], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, col(c2 * 1.25 + 15), 1, cv2.LINE_AA)
    cx = int(w * rng.choice([0.25, 0.75]))
    cv2.ellipse(img, (cx, h // 2), (int(h * 0.30), int(h * 0.38)), 0, 0, 360, col(c1 * 0.72), -1, cv2.LINE_AA)
    cv2.putText(img, str(int(rng.choice([1000, 5000, 20000, 50000, 100000, 500000]))), (int(w * 0.06), int(h * 0.25)),
                cv2.FONT_HERSHEY_DUPLEX, h / 140.0, col(c1 * 0.45), 2, cv2.LINE_AA)
    for r in range(int(rng.integers(2, 5))):
        y = int(h * (0.62 + 0.09 * r))
        for x in range(int(w * 0.05), int(w * 0.5), 14):
            cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 11)), y + 4), col(c1 * 0.5), -1)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img


def _rect_corners(cx, cy, w, h, angle_deg, jitter):
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    base = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32)
    return (base @ rot.T + np.array([cx, cy], np.float32) + jitter).astype(np.float32)


def _regions(n: int):
    if n == 1:
        return [(0, 0, int(W * 0.62), H)], (int(W * 0.66), 80, W - 30, H - 80)   # vùng tờ, vùng vật gây nhiễu
    if n == 2:
        return [(0, 0, W // 2, H), (W // 2, 0, W, H)], None
    return [(0, 0, W // 2, H // 2), (W // 2, 0, W, H // 2), (0, H // 2, W, H)], None


class Sequence:
    """Một cảnh có nhiều khung; mọi thứ cố định theo seed."""

    def __init__(self, seed: int, speed: float, hard: bool, hand: bool = False):
        rng = self.rng = np.random.default_rng(seed)
        self.speed, self.hard, self.hand = speed, hard, hand
        self.bg = make_background(rng)
        if hand:
            self._init_hand(rng)
            return
        bg_mean = self.bg.reshape(-1, 3).mean(axis=0)
        n = int(rng.choice([1, 1, 2, 3]))
        regions, distractor_box = _regions(n)
        self.notes = []
        for i, (x0, y0, x1, y1) in enumerate(regions):
            long_side = int(rng.uniform(380, 520) if n == 1 else rng.uniform(250, 340) if n == 2 else rng.uniform(210, 270))
            ratio = float(rng.uniform(1.95, 2.35))
            low = bool(hard and rng.random() < 0.5)
            angle = float(rng.uniform(-40, 40) if rng.random() < 0.85 else rng.choice([-1, 1]) * rng.uniform(80, 100))
            jit = rng.normal(0, 0.012 * long_side, (4, 2)).astype(np.float32)       # phối cảnh nhẹ
            self.notes.append(dict(
                region=(x0, y0, x1, y1), w=long_side, h=long_side / ratio, angle=angle, jit=jit,
                c=np.array([(x0 + x1) / 2 + rng.uniform(-30, 30), (y0 + y1) / 2 + rng.uniform(-30, 30)], np.float32),
                v=np.array([rng.normal(), rng.normal()], np.float32) * speed,
                om=float(rng.uniform(-0.3, 0.3) * speed / 2),
                tex=make_texture(rng, long_side, ratio, bg_mean if low else None),
                finger=(int(rng.integers(0, 4)) if (hard and rng.random() < 0.5) else None)))
        self.distractor = None
        if distractor_box is not None and rng.random() < 0.6:
            x0, y0, x1, y1 = distractor_box
            self.distractor = dict(c=((x0 + x1) / 2, (y0 + y1) / 2), w=min(240, x1 - x0 - 20), ratio=float(rng.uniform(1.5, 1.8)),
                                   angle=float(rng.uniform(-30, 30)), col=tuple(int(v) for v in rng.integers(20, 235, 3)))
        self.gain = float(rng.uniform(0.85, 1.15))
        self.noise = float(rng.choice([1.5, 3.0, 5.0]))
        self.blur_len = int(max(0, speed * 0.9))

    def _init_hand(self, rng) -> None:
        """Cảnh CẦM TAY: một tờ bị ngón tay/bàn tay che một phần, phía sau là vật chữ nhật lớn (màn hình, hộp, sách)."""
        bg = self.bg.astype(np.float32)
        for _ in range(int(rng.integers(2, 4))):                   # vật lớn phía sau: cạnh thẳng, rõ nét
            w = int(rng.uniform(380, 640)); h = int(w / rng.uniform(1.3, 1.9))
            c = (rng.uniform(w / 2, W - w / 2), rng.uniform(h / 2, H - h / 2))
            q = _rect_corners(c[0], c[1], w, h, float(rng.uniform(-12, 12)), 0)
            col = tuple(int(v) for v in rng.choice([[35, 35, 40], [225, 225, 225], [60, 90, 140], [140, 150, 120]]))
            cv2.fillConvexPoly(bg, np.round(q).astype(np.int32), col, cv2.LINE_AA)
            cv2.polylines(bg, [np.round(q).astype(np.int32)], True, (20, 20, 20), 3, cv2.LINE_AA)
            inner = _rect_corners(c[0], c[1], w * 0.86, h * 0.8, 0, 0)
            cv2.fillConvexPoly(bg, np.round(inner).astype(np.int32), tuple(int(255 - v * 0.6) for v in col), cv2.LINE_AA)
        self.bg = np.clip(bg, 0, 255).astype(np.uint8)
        bg_mean = self.bg.reshape(-1, 3).mean(axis=0)
        long_side = int(rng.uniform(400, 540)); ratio = float(rng.uniform(1.95, 2.35))
        self.notes = [dict(
            region=(160, 60, W - 160, H - 60), w=long_side, h=long_side / ratio, angle=float(rng.uniform(-35, 35)),
            jit=rng.normal(0, 0.01 * long_side, (4, 2)).astype(np.float32),
            c=np.array([W / 2 + rng.uniform(-120, 120), H / 2 + rng.uniform(-60, 60)], np.float32),
            v=np.array([rng.normal(), rng.normal()], np.float32) * self.speed,
            om=float(rng.uniform(-0.3, 0.3) * self.speed / 2),
            tex=make_texture(rng, long_side, ratio, bg_mean if rng.random() < 0.25 else None), finger=None)]
        self.hand_side = int(rng.integers(0, 4))                   # cạnh nào được cầm
        self.distractor = None
        self.gain = float(rng.uniform(0.85, 1.15)); self.noise = float(rng.choice([1.5, 3.0, 5.0]))
        self.blur_len = int(max(0, self.speed * 0.9))

    def _draw_hand(self, out: np.ndarray, g: np.ndarray, angle: float) -> None:
        k = self.hand_side
        a, b = g[k], g[(k + 1) % 4]
        cen = g.mean(axis=0)
        edge = (b - a) / (np.linalg.norm(b - a) + 1e-6)
        inward = cen - (a + b) / 2; inward /= np.linalg.norm(inward) + 1e-6
        for f in (0.2, 0.4, 0.6, 0.8):                             # bốn ngón đè lên mép tờ ~22 px
            p = a + (b - a) * f + inward * 6
            cv2.ellipse(out, (int(p[0]), int(p[1])), (24, 36), float(np.degrees(np.arctan2(edge[1], edge[0])) + 90), 0, 360, SKIN, -1, cv2.LINE_AA)
        t = g[(k + 2) % 4] + (cen - g[(k + 2) % 4]) * 0.05            # ngón cái đè lên một góc đối diện
        cv2.ellipse(out, (int(t[0]), int(t[1])), (40, 24), float(angle), 0, 360, SKIN, -1, cv2.LINE_AA)
        mid = (a + b) / 2 - inward * 140                            # bàn tay/cổ tay kéo ra ngoài tới mép khung
        palm = np.array([a - inward * 20, b - inward * 20, b - inward * 260 + edge * 40, a - inward * 260 - edge * 40])
        cv2.fillConvexPoly(out, np.round(palm).astype(np.int32), SKIN, cv2.LINE_AA)

    # --------------------------------------------------------------
    def _quad(self, nt) -> np.ndarray:
        return _rect_corners(nt["c"][0], nt["c"][1], nt["w"], nt["h"], nt["angle"], nt["jit"])

    def _step(self, nt):
        nt["c"] = nt["c"] + nt["v"]
        nt["angle"] += nt["om"]
        x0, y0, x1, y1 = nt["region"]
        q = self._quad(nt)
        m = 14
        for axis, lo, hi in ((0, x0 + m, x1 - m), (1, y0 + m, y1 - m)):
            if q[:, axis].min() < lo:
                nt["c"][axis] += lo - q[:, axis].min(); nt["v"][axis] = abs(nt["v"][axis])
            elif q[:, axis].max() > hi:
                nt["c"][axis] -= q[:, axis].max() - hi; nt["v"][axis] = -abs(nt["v"][axis])

    def frame(self, t: int):
        """Trả (ảnh BGR, danh sách 4 góc đáp án đã sắp bằng order_corners)."""
        from app.core.note_detector import order_corners
        if t > 0:
            for nt in self.notes:
                self._step(nt)
        out = self.bg.astype(np.float32)
        gts = []
        for nt in self.notes:
            quad = self._quad(nt)
            gts.append(order_corners(quad))
            th, tw = nt["tex"].shape[:2]
            M = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), quad)
            warped = cv2.warpPerspective(nt["tex"], M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
            mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
            shadow = cv2.GaussianBlur(np.roll(np.roll(mask, 6, 0), 4, 1), (0, 0), 5)[..., None]
            a = mask[..., None]
            out = out * (1 - 0.30 * shadow * (1 - a))
            out = out * (1 - a) + warped * a
        if self.distractor:
            d = self.distractor
            q = _rect_corners(d["c"][0], d["c"][1], d["w"], d["w"] / d["ratio"], d["angle"], 0)
            cv2.fillConvexPoly(out, np.round(q).astype(np.int32), d["col"], cv2.LINE_AA)
            cv2.polylines(out, [np.round(q).astype(np.int32)], True, (40, 40, 40), 2, cv2.LINE_AA)
        if self.hand:
            self._draw_hand(out, gts[0], self.notes[0]["angle"])
        for nt, g in zip(self.notes, gts):               # ngón tay che một góc
            if nt["finger"] is not None:
                k = nt["finger"]
                cen = g.mean(axis=0)
                direction = (g[k] - cen) / (np.linalg.norm(g[k] - cen) + 1e-6)
                p = g[k] + direction * 4
                cv2.ellipse(out, (int(p[0]), int(p[1])), (44, 27), float(nt["angle"]), 0, 360, SKIN, -1, cv2.LINE_AA)
        out = out * self.gain
        if self.blur_len >= 2:                           # mờ chuyển động theo hướng di chuyển
            v = self.notes[0]["v"]
            k = np.zeros((self.blur_len, self.blur_len), np.float32)
            ang = np.arctan2(v[1], v[0])
            c = self.blur_len // 2
            cv2.line(k, (int(c - np.cos(ang) * c), int(c - np.sin(ang) * c)),
                     (int(c + np.cos(ang) * c), int(c + np.sin(ang) * c)), 1.0)
            out = cv2.filter2D(out, -1, k / max(k.sum(), 1.0))
        out = out + self.rng.normal(0, self.noise, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8), gts


# ----------------------------------------------------------------------------
# Đo
# ----------------------------------------------------------------------------
def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    ua = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / max(ua, 1e-6))


def corner_error(det: np.ndarray, gt: np.ndarray) -> tuple[float, int]:
    best = min(range(4), key=lambda k: np.linalg.norm(det - np.roll(gt, -k, axis=0), axis=1).mean())
    return float(np.linalg.norm(det - np.roll(gt, -best, axis=0), axis=1).mean()), best


def match(gts, dets):
    pairs = sorted(((poly_iou(g, d), i, j) for i, g in enumerate(gts) for j, d in enumerate(dets)), reverse=True)
    used_g, used_d, out = set(), set(), {}
    for iou, i, j in pairs:
        if iou < 0.3 or i in used_g or j in used_d:
            continue
        used_g.add(i); used_d.add(j); out[i] = (j, iou)
    return out


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Stats:
    def __init__(self):
        self.gt = self.found = self.fp = self.frames = 0
        self.ious, self.errs, self.ms = [], [], []
        self.prev_found = self.lost = 0
        self.jitter = []


def run(detector_modules: dict, n_seqs: int, n_frames: int, seed: int, work_width: int, n_hand: int = 0):
    stats = {name: {"all": Stats(), "static": Stats(), "fast": Stats(), "hard": Stats(), "hand": Stats()} for name in detector_modules}
    speeds = [0, 0, 2.5, 6, 12]
    t_start = time.time()
    for s in range(n_seqs + n_hand):
        hand = s >= n_seqs
        speed = (0, 2.5, 6)[s % 3] if hand else speeds[s % len(speeds)]
        hard = False if hand else (s // len(speeds)) % 2 == 1
        seq = Sequence(seed * 1000 + s, speed, hard, hand)
        dets_obj = {}
        for name, mod in detector_modules.items():
            dets_obj[name] = mod.ClassicNoteDetector(mod.ClassicParams(max_notes=5, work_width=work_width))
        track = {name: {} for name in detector_modules}      # (note_idx) -> lịch sử vị trí góc đã căn theo đáp án
        prev_ok = {name: set() for name in detector_modules}
        for t in range(n_frames):
            frame, gts = seq.frame(t)
            for name, det in dets_obj.items():
                t0 = time.perf_counter()
                dets = det.detect(frame)
                ms = (time.perf_counter() - t0) * 1000
                m = match(gts, [d.corners for d in dets])
                if hand:
                    groups = [stats[name]["hand"]]
                else:
                    groups = [stats[name]["all"]] + ([stats[name]["static"]] if speed == 0 else []) \
                        + ([stats[name]["fast"]] if speed >= 6 else []) + ([stats[name]["hard"]] if hard else [])
                ok_now = set()
                for st in groups:
                    st.frames += 1; st.ms.append(ms); st.gt += len(gts); st.fp += len(dets) - len(m)
                for i, (j, iou) in m.items():
                    err, shift = corner_error(dets[j].corners, gts[i])
                    good = iou >= 0.7
                    for st in groups:
                        st.ious.append(iou); st.errs.append(err)
                        st.found += int(good)
                    if good:
                        ok_now.add(i)
                    if speed == 0 and good and not hand:       # góc đã xoay vòng cho khớp thứ tự đáp án
                        track[name].setdefault(i, []).append(np.roll(dets[j].corners, shift, axis=0))
                for st in groups:
                    st.prev_found += len(prev_ok[name]); st.lost += len(prev_ok[name] - ok_now)
                prev_ok[name] = ok_now
        for name in detector_modules:                         # rung khi đứng yên
            if speed == 0 and not hand:
                for pts in track[name].values():
                    if len(pts) >= 5:
                        j = float(np.mean(np.std(np.stack(pts), axis=0)))   # độ lệch chuẩn của từng góc qua các khung
                        for key in ("all", "static"):
                            stats[name][key].jitter.append(j)
        print(f"\r  chuỗi {s + 1}/{n_seqs + n_hand}  ({time.time() - t_start:0.0f}s)", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return stats


def summarize(st: Stats) -> dict:
    return {
        "frames": st.frames,
        "recall %": 100.0 * st.found / max(st.gt, 1),
        "iou": float(np.mean(st.ious)) if st.ious else 0.0,
        "err_px": float(np.mean(st.errs)) if st.errs else 0.0,
        "err_p90": float(np.percentile(st.errs, 90)) if st.errs else 0.0,
        "fp/khung": st.fp / max(st.frames, 1),
        "rot_khung %": 100.0 * st.lost / max(st.prev_found, 1),
        "rung_px": float(np.mean(st.jitter)) if st.jitter else float("nan"),
        "ms": float(np.mean(st.ms)) if st.ms else 0.0,
        "ms_p95": float(np.percentile(st.ms, 95)) if st.ms else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", type=int, default=20)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hand", type=int, default=12, help="số chuỗi CẦM TAY (tờ bị ngón tay che, vật lớn phía sau)")
    ap.add_argument("--width", type=int, default=480, help="độ rộng làm việc của bộ tìm (như DETECT_WORK_WIDTH)")
    ap.add_argument("--compare", default=None, help="đường dẫn một note_detector.py khác để so sánh cạnh nhau")
    args = ap.parse_args()

    mods = {"hien_tai": load_module(ROOT / "app" / "core" / "note_detector.py", "nd_current")}
    if args.compare:
        mods = {"so_sanh": load_module(Path(args.compare), "nd_compare"), **mods}
    stats = run(mods, args.seqs, args.frames, args.seed, args.width, args.hand)

    names = list(mods)
    for group, title in (("all", "TẤT CẢ"), ("static", "TỜ ĐỨNG YÊN"), ("fast", "DI CHUYỂN NHANH (>=6 px/khung)"),
                         ("hard", "KHÓ: tương phản thấp / ngón tay che góc"),
                         ("hand", "CẦM TAY: ngón tay che mép + vật lớn phía sau")):
        if not stats[names[0]][group].frames:
            continue
        rows = {n: summarize(stats[n][group]) for n in names}
        print(f"\n== {title}  ({rows[names[0]]['frames']} khung) ==")
        print(f"{'chỉ số':14s}" + "".join(f"{n:>12s}" for n in names))
        for k in ("recall %", "iou", "err_px", "err_p90", "fp/khung", "rot_khung %", "rung_px", "ms", "ms_p95"):
            print(f"{k:14s}" + "".join(f"{rows[n][k]:12.3f}" for n in names))


if __name__ == "__main__":
    main()
