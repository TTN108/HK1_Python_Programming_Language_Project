"""Bộ phân loại mệnh giá bằng ONNX (MobileNetV3).

Hợp đồng:
    đầu vào : ảnh một tờ đã nắn phẳng, uint8, HxWx3, thứ tự kênh RGB (mặc định) - tự thu về 224x224
    đầu ra  : ClassifyResult(label, confidence, status, value, ...)
              status = "ok"        -> mệnh giá đủ tin cậy, được cộng tiền (value > 0)
              status = "uncertain" -> độ tin cậy < CONF_THRESHOLD, chưa được cộng (cần sửa tay)
              status = "not_money" -> lớp khong_phai_tien, không được cộng tiền

Chuẩn hoá ảnh phải khớp lúc train: RGB, /255, trừ mean / chia std ImageNet.

Thử nhanh:
    python classifier.py anh_to_tien1.jpg anh_to_tien2.jpg
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

import config

STATUS_OK = "ok"
STATUS_UNCERTAIN = "uncertain"
STATUS_NOT_MONEY = "not_money"


@dataclass(frozen=True)
class ClassifyResult:
    label: str                         # lớp dự đoán (luôn có, kể cả khi uncertain)
    confidence: float                  # xác suất của lớp dự đoán, 0..1
    status: str                        # "ok" | "uncertain" | "not_money"
    value: int                         # số tiền (đồng); CHỈ > 0 khi status == "ok"
    runner_up: tuple[str, float]       # lớp đứng thứ hai và xác suất (để đánh giá độ lưỡng lự)
    probs: np.ndarray = field(repr=False, compare=False)  # xác suất mọi lớp, theo thứ tự CLASSES

    @property
    def counted(self) -> bool:
        return self.status == STATUS_OK


class CurrencyClassifier:
    def __init__(
        self,
        model_path: str | Path | None = None,
        classes: Sequence[str] | None = None,
        conf_threshold: float | None = None,
        use_tta: bool | None = None,
        tta_skip_conf: float | None = None,
        providers: Sequence[str] | None = None,
    ):
        self.model_path = Path(model_path or config.CLASSIFIER_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không thấy mô hình phân loại: {self.model_path}")
        if self.model_path.stat().st_size == 0:
            raise ValueError(f"File mô hình rỗng (chưa chép mô hình thật vào): {self.model_path}")

        self.classes = list(classes if classes is not None else config.CLASSES)
        self.conf_threshold = config.CONF_THRESHOLD if conf_threshold is None else conf_threshold
        self.use_tta = config.USE_TTA if use_tta is None else use_tta
        self.tta_skip_conf = config.TTA_SKIP_CONF if tta_skip_conf is None else tta_skip_conf
        self._check_classes()

        opts = ort.SessionOptions()
        if config.ONNX_THREADS:
            opts.intra_op_num_threads = config.ONNX_THREADS
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=list(providers or self._pick_providers())
        )

        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        self._dtype = np.float16 if "float16" in str(inp.type) else np.float32
        shape = list(inp.shape)
        if len(shape) != 4 or (isinstance(shape[1], int) and shape[1] != 3):
            raise ValueError(f"Mô hình cần đầu vào NCHW với 3 kênh, nhưng có shape {shape}")
        # Mô hình xuất với batch cố định = 1 thì phải chạy từng ảnh một.
        self._single = isinstance(shape[0], int)
        if self._single and shape[0] != 1:
            raise ValueError(f"Chỉ hỗ trợ batch động hoặc batch = 1, mô hình có batch {shape[0]}")
        self.size = int(shape[2]) if isinstance(shape[2], int) else config.WARP_SIZE

        out_shape = self.session.get_outputs()[0].shape
        if isinstance(out_shape[-1], int) and out_shape[-1] != len(self.classes):
            raise ValueError(
                f"Mô hình có {out_shape[-1]} lớp nhưng danh sách CLASSES có {len(self.classes)} lớp. "
                "Sửa config.CLASSES hoặc tạo models/classes.json đúng thứ tự lúc train."
            )

        self._mean = np.array(config.IMAGENET_MEAN, np.float32).reshape(1, 1, 3)
        self._std = np.array(config.IMAGENET_STD, np.float32).reshape(1, 1, 3)
        self.last_ms = 0.0  # thời gian phân loại của lần gọi gần nhất (hiện trên thanh trạng thái)

    # ------------------------------------------------------------------
    def _check_classes(self) -> None:
        bad = [c for c in self.classes if c != config.NOT_MONEY_CLASS and config.class_to_value(c) <= 0]
        if bad:
            raise ValueError(f"Các lớp này không đọc được thành số tiền: {bad}")

    @staticmethod
    def _pick_providers() -> list[str]:
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _preprocess(self, img: np.ndarray, bgr: bool) -> np.ndarray:
        if img is None or img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError("Ảnh tờ phải là mảng uint8 dạng HxWx3")
        if bgr:
            img = np.ascontiguousarray(img[..., ::-1])
        if img.shape[0] != self.size or img.shape[1] != self.size:
            shrink = img.shape[0] > self.size or img.shape[1] > self.size
            img = cv2.resize(img, (self.size, self.size),
                             interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0 - self._mean) / self._std
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    @staticmethod
    def _to_probs(out: np.ndarray) -> np.ndarray:
        """Đầu ra đã là xác suất thì giữ nguyên, ngược lại coi là logit và softmax."""
        if (out >= 0).all() and np.allclose(out.sum(axis=1), 1.0, atol=1e-3):
            return out
        e = np.exp(out - out.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        """(N,3,S,S) -> (N,C) xác suất."""
        batch = batch.astype(self._dtype, copy=False)
        if self._single:
            out = np.concatenate(
                [np.asarray(self.session.run(None, {self._input_name: b[None]})[0]) for b in batch], axis=0
            )
        else:
            out = np.asarray(self.session.run(None, {self._input_name: batch})[0])
        out = out.astype(np.float32).reshape(len(batch), -1)
        if out.shape[1] != len(self.classes):
            raise ValueError(f"Mô hình trả {out.shape[1]} lớp, CLASSES có {len(self.classes)} lớp")
        return self._to_probs(out)

    def _make_result(self, p: np.ndarray) -> ClassifyResult:
        order = np.argsort(p)[::-1]
        top = int(order[0])
        label, conf = self.classes[top], float(p[top])
        second = int(order[1]) if len(order) > 1 else top
        if label == config.NOT_MONEY_CLASS:
            status = STATUS_NOT_MONEY
        elif conf < self.conf_threshold:
            status = STATUS_UNCERTAIN
        else:
            status = STATUS_OK
        value = config.class_to_value(label) if status == STATUS_OK else 0
        return ClassifyResult(label, conf, status, value, (self.classes[second], float(p[second])), p)

    # ------------------------------------------------------------------
    def predict_batch(self, images: Sequence[np.ndarray], bgr: bool = False) -> list[ClassifyResult]:
        """Phân loại nhiều tờ một lần (tối đa vài tờ mỗi khung hình).

        bgr=True nếu ảnh đang ở thứ tự kênh BGR của OpenCV.
        """
        if len(images) == 0:
            return []
        t0 = time.perf_counter()
        x = np.stack([self._preprocess(im, bgr) for im in images])
        probs = self._run(x)
        if self.use_tta:
            weak = [i for i in range(len(x)) if probs[i].max() < self.tta_skip_conf]
            if weak:  # chỉ tờ kém chắc chắn mới chạy thêm lượt xoay 180° rồi lấy trung bình
                rotated = np.ascontiguousarray(x[weak][:, :, ::-1, ::-1])
                probs[weak] = (probs[weak] + self._run(rotated)) / 2.0
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return [self._make_result(p) for p in probs]

    def predict(self, image: np.ndarray, bgr: bool = False) -> ClassifyResult:
        return self.predict_batch([image], bgr=bgr)[0]

    def warmup(self) -> None:
        """Chạy thử một ảnh trống để lần phân loại đầu tiên không bị chậm."""
        self.predict(np.zeros((self.size, self.size, 3), np.uint8))


# ----------------------------------------------------------------------
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Thử bộ phân loại trên ảnh tờ tiền đã cắt/nắn phẳng")
    ap.add_argument("images", nargs="+", help="đường dẫn ảnh tờ tiền")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    clf = CurrencyClassifier(args.model, conf_threshold=args.conf, use_tta=False if args.no_tta else None)
    clf.warmup()
    for path in args.images:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # đọc được đường dẫn có dấu
        if img is None:
            print(f"{path}: không đọc được ảnh")
            continue
        r = clf.predict(img, bgr=True)  # OpenCV đọc ra BGR
        print(f"{path}: {r.label}  {r.confidence:.1%}  [{r.status}]  "
              f"(thứ hai: {r.runner_up[0]} {r.runner_up[1]:.1%})  {clf.last_ms:.0f} ms")


if __name__ == "__main__":
    _main()