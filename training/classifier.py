# training/classifier.py
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
import onnxruntime as ort
from training.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNCERTAIN_LABEL,
)


@dataclass
class ClassificationResult:
    label: str          # Mệnh giá ("1000".."500000"), "khong_phai_tien", hoặc "không chắc chắn"
    confidence: float   # Độ tin cậy [0.0, 1.0]
    raw_label: str      # Nhãn có xác suất cao nhất trước khi áp ngưỡng


class BillClassifier:
    def __init__(
        self,
        onnx_path: str = "models/vnd_classifier_v1.onnx",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def _preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        if bgr_img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            bgr_img = cv2.resize(bgr_img, (IMAGE_SIZE, IMAGE_SIZE))
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self._mean) / self._std
        return np.transpose(norm, (2, 0, 1))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict(self, bill_bgr_224: np.ndarray) -> ClassificationResult:
        return self.predict_batch([bill_bgr_224])[0]

    def predict_batch(self, bills_bgr_list: List[np.ndarray]) -> List[ClassificationResult]:
        if not bills_bgr_list:
            return []
        batch = np.stack([self._preprocess(img) for img in bills_bgr_list], axis=0)
        logits = self.session.run(None, {self.input_name: batch})[0]
        probs = self._softmax(logits)

        results: List[ClassificationResult] = []
        for p in probs:
            idx = int(np.argmax(p))
            conf = float(p[idx])
            raw_lbl = IDX_TO_CLASS[idx]
            final_lbl = UNCERTAIN_LABEL if conf < self.confidence_threshold else raw_lbl
            results.append(ClassificationResult(label=final_lbl, confidence=conf, raw_label=raw_lbl))
        return results