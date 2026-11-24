# src/model/export_onnx.py
import argparse
import os
import numpy as np
import onnx
import onnxruntime as ort
import torch
from src.model.config import IMAGE_SIZE
from src.model.train import build_model


def export_to_onnx(checkpoint: str, backbone: str, output_path: str) -> None:
    model = build_model(backbone)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx.checker.check_model(onnx.load(output_path))

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    ort_sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_out = ort_sess.run(None, {"input": dummy.numpy()})[0]

    print(f"Xuất ONNX thành công: {output_path}")
    print(f"Sai số lớn nhất (PyTorch vs ONNX): {np.max(np.abs(torch_out - ort_out)):.6e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--output", type=str, default="models/vnd_classifier_v1.onnx")
    args = p.parse_args()
    export_to_onnx(args.checkpoint, args.backbone, args.output)