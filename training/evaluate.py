# training/evaluate.py
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from training.classifier import BillClassifier
from training.config import CLASS_NAMES, CLASS_TO_IDX


def evaluate(split_dir: str, onnx_path: str, save_cm_path: str) -> None:
    clf = BillClassifier(onnx_path=onnx_path)
    y_true, y_pred, confs = [], [], []

    for cname in CLASS_NAMES:
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir):
            continue
        t_idx = CLASS_TO_IDX[cname]
        for fname in os.listdir(cdir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(cdir, fname))
            if img is None:
                continue
            res = clf.predict(img)
            y_true.append(t_idx)
            y_pred.append(CLASS_TO_IDX[res.raw_label])
            confs.append(res.confidence)

    y_true_arr, y_pred_arr, conf_arr = np.array(y_true), np.array(y_pred), np.array(confs)

    print("=== CLASSIFICATION REPORT (TRƯỚC ÁP NGƯỠNG) ===")
    print(classification_report(
        y_true_arr, y_pred_arr,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4, zero_division=0
    ))

    os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Nhận diện tiền VND")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Thực tế")
    plt.xlabel("Dự đoán")
    plt.tight_layout()
    plt.savefig(save_cm_path, dpi=300)
    print(f"Đã lưu Confusion Matrix tại: {save_cm_path}")

    print("\n=== KHẢO SÁT NGƯỠNG TIN CẬY ===")
    print(f"{'Ngưỡng':<8} | {'Sang không chắc chắn (<10%)':<28} | {'Sai nhưng tự tin (<1%)':<25}")
    print("-" * 68)
    for th in [0.80, 0.85, 0.90, 0.92, 0.95]:
        mask = conf_arr >= th
        uncertain_pct = np.mean(~mask) * 100.0
        wrong_conf_pct = np.mean(y_pred_arr[mask] != y_true_arr[mask]) * 100.0 if np.any(mask) else 0.0
        print(f"{th:<8.2f} | {uncertain_pct:<27.2f}% | {wrong_conf_pct:<24.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", type=str, default="data_v1/val")
    p.add_argument("--onnx_path", type=str, default="models/vnd_classifier_v1.onnx")
    p.add_argument("--save_cm", type=str, default="logs/confusion_matrix.png")
    args = p.parse_args()
    evaluate(args.split_dir, args.onnx_path, args.save_cm)