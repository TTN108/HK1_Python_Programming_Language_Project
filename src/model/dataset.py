# src/model/dataset.py
import os
from typing import Callable, List, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from src.model.config import CLASS_NAMES, CLASS_TO_IDX, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=(180, 180), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(
            scale=(0.92, 1.0),
            rotate=(-10, 10),
            translate_percent=(-0.03, 0.03),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(16, 40),
            hole_width_range=(16, 40),
            fill=0,
            p=0.35,
        ),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.25),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 1),
            src_radius=60,
            src_color=(255, 255, 255),
            num_flare_circles_range=(1, 2),
            p=0.15,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class VNDBillDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = CLASS_TO_IDX[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        if not self.samples:
            raise RuntimeError(f"Không tìm thấy ảnh hợp lệ trong: {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Lỗi đọc ảnh: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            tensor_img = self.transform(image=img_rgb)["image"]
        else:
            tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        return tensor_img, label

    def get_targets(self) -> List[int]:
        return [label for _, label in self.samples]


def create_balanced_sampler(dataset: VNDBillDataset) -> WeightedRandomSampler:
    targets = dataset.get_targets()
    class_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )