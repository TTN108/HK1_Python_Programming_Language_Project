# training/prepare_data.py
import argparse
import os
from pathlib import Path
import cv2
import yaml
from roboflow import Roboflow
from tqdm import tqdm
from training.config import CLASS_NAMES, IMAGE_SIZE

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
BG_LIMITS_PER_DATASET = {"train": 200, "val": 50, "test": 50}


def download_datasets(api_key: str, raw_root: Path) -> None:
    rf = Roboflow(api_key=api_key)

    ds1_dir = raw_root / "raw_data_1"
    if not (ds1_dir / "data.yaml").exists():
        print("Đang tải Nguồn 1 (first-qkz37/vietnamese-currency-lgi9i v6)...")
        project1 = rf.workspace("first-qkz37").project("vietnamese-currency-lgi9i")
        project1.version(6).download("yolov8", location=str(ds1_dir))

    ds2_dir = raw_root / "raw_data_2"
    if not (ds2_dir / "data.yaml").exists():
        print("Đang tải Nguồn 2 (tran-minh-nghia/vietnam-money-recognize-dy4cd-iddjh v1)...")
        project2 = rf.workspace("tran-minh-nghia").project("vietnam-money-recognize-dy4cd-iddjh")
        project2.version(1).download("yolov8", location=str(ds2_dir))


def process_yolo_dataset(raw_dir: Path, output_root: Path, prefix: str) -> None:
    yaml_path = raw_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"Không tìm thấy {yaml_path}. Bỏ qua {raw_dir}.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        class_names = [str(c) for c in yaml.safe_load(f)["names"]]

    print(f"\nĐang xử lý '{raw_dir.name}' - Các lớp: {class_names}")
    count_notes = 0
    total_bg = 0

    for rf_split, our_split in SPLIT_MAP.items():
        img_dir = raw_dir / rf_split / "images"
        lbl_dir = raw_dir / rf_split / "labels"
        if not img_dir.exists():
            continue

        count_bg_split = 0
        max_bg_split = BG_LIMITS_PER_DATASET[our_split]

        for img_path in tqdm(list(img_dir.glob("*.*")), desc=f"[{prefix} -> {our_split}]"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w, _ = img.shape

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cname = class_names[cls_id]
                if cname not in CLASS_NAMES:
                    continue

                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = max(0, int((xc - bw / 2) * w))
                y1 = max(0, int((yc - bh / 2) * h))
                x2 = min(w, int((xc + bw / 2) * w))
                y2 = min(h, int((yc + bh / 2) * h))

                crop = img[y1:y2, x1:x2]
                if crop.size == 0 or (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                crop_224 = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
                save_path = output_root / our_split / cname / f"{cname}_{prefix}_{img_path.stem}_{idx}.jpg"
                cv2.imwrite(str(save_path), crop_224)
                count_notes += 1

                # Cắt mẫu nền "khong_phai_tien" chia đều cho cả train, val, test
                if count_bg_split < max_bg_split and y1 > 100 and x1 > 100:
                    bg_crop = cv2.resize(img[0:y1, 0:x1], (IMAGE_SIZE, IMAGE_SIZE))
                    bg_path = output_root / our_split / "khong_phai_tien" / f"bg_{prefix}_{img_path.stem}_{idx}.jpg"
                    cv2.imwrite(str(bg_path), bg_crop)
                    count_bg_split += 1
                    total_bg += 1

    print(f"Hoàn tất {raw_dir.name}: {count_notes} ảnh tiền + {total_bg} ảnh 'khong_phai_tien'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải và chuẩn bị dữ liệu tiền VND từ Roboflow")
    parser.add_argument("--raw_root", type=str, default="raw_datasets")
    parser.add_argument("--output_dir", type=str, default="data_v1")
    parser.add_argument("--skip_download", action="store_true", help="Bỏ qua bước tải nếu đã có thư mục raw")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_dir)

    for split in ["train", "val", "test"]:
        for cname in CLASS_NAMES:
            (output_root / split / cname).mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Chưa thiết lập biến môi trường ROBOFLOW_API_KEY. "
                "Chạy lệnh: $env:ROBOFLOW_API_KEY='your_key' (PowerShell) trước khi chạy script."
            )
        download_datasets(api_key, raw_root)

    process_yolo_dataset(raw_root / "raw_data_1", output_root, "ds1")
    process_yolo_dataset(raw_root / "raw_data_2", output_root, "ds2")


if __name__ == "__main__":
    main()