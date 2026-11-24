"""Liệt kê các mô hình ONNX trong models/ cùng thứ tự lớp của từng mô hình.

Mỗi mô hình = một file .onnx. Thứ tự lớp đọc từ file đi kèm <tên>.classes.json
(do `python -m training.export_onnx` ghi). Mô hình không có file này vẫn dùng được,
nhưng thứ tự lớp là SUY ĐOÁN (giao diện sẽ cảnh báo) — nếu sai, tiền sẽ bị nhận nhầm mệnh giá.

Thử nhanh:  python -m app.core.model_registry
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app import config

SOURCE_TEXT = {"sidecar": "file lớp riêng", "shared": "classes.json dùng chung", "default": "thứ tự mặc định"}


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    classes: tuple[str, ...]
    classes_source: str          # "sidecar" | "shared" | "default"

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / 1e6

    @property
    def trusted(self) -> bool:
        """Thứ tự lớp có file riêng đảm bảo?"""
        return self.classes_source == "sidecar"


def describe(path: Path | str) -> ModelInfo:
    path = Path(path)
    classes, source = config.classes_for_model(path)
    return ModelInfo(path, tuple(classes), source)


def discover(models_dir: Path | None = None) -> list[ModelInfo]:
    """Mọi .onnx không rỗng trong models/, mô hình mặc định (config.CLASSIFIER_PATH) đứng đầu."""
    folder = Path(models_dir or config.MODELS_DIR)
    paths = sorted(p for p in folder.glob("*.onnx") if p.stat().st_size > 0)
    paths.sort(key=lambda p: p.resolve() != Path(config.CLASSIFIER_PATH).resolve())
    infos = []
    for p in paths:
        try:
            infos.append(describe(p))
        except (ValueError, KeyError, OSError) as exc:     # file lớp hỏng: bỏ qua mô hình này, không làm sập app
            print(f"[model_registry] bỏ qua {p.name}: {exc}")
    return infos


if __name__ == "__main__":
    for m in discover():
        print(f"{m.name:32s} {m.size_mb:5.1f} MB  {len(m.classes)} lớp  [{SOURCE_TEXT[m.classes_source]}]  {list(m.classes)}")
