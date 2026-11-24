# src/model/train.py
import argparse
import csv
import os
from datetime import datetime
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from src.model.config import CLASS_NAMES
from src.model.dataset import (
    VNDBillDataset,
    create_balanced_sampler,
    get_train_transforms,
    get_val_transforms,
)


def build_model(backbone_name: str, num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    if backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone_name}")
    return model


def freeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 3) -> None:
    total = len(model.features)
    for idx in range(max(0, total - num_blocks), total):
        for p in model.features[idx].parameters():
            p.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def log_experiment(log_csv: str, row_data: list) -> None:
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    exists = os.path.exists(log_csv)
    with open(log_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp", "data_version", "backbone", "batch_size",
                "lr_stage1", "lr_stage2", "label_smoothing",
                "best_val_loss", "best_val_acc", "checkpoint"
            ])
        writer.writerow(row_data)


def train_pipeline(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")

    train_ds = VNDBillDataset(os.path.join(args.data_dir, "train"), transform=get_train_transforms())
    val_ds = VNDBillDataset(os.path.join(args.data_dir, "val"), transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=create_balanced_sampler(train_ds), num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, f"best_{args.backbone}_{args.data_version}.pth")
    best_val_loss, best_val_acc = float("inf"), 0.0

    # Giai đoạn 1: Đóng băng backbone
    print(f"\n--- GIAI ĐOẠN 1 ({args.epochs_stage1} epochs) ---")
    freeze_backbone(model)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_stage1, weight_decay=1e-2)
    for ep in range(1, args.epochs_stage1 + 1):
        tr_l, tr_a = run_epoch(model, train_loader, criterion, opt1, device)
        va_l, va_a = run_epoch(model, val_loader, criterion, None, device)
        print(f"[GĐ1 {ep:02d}/{args.epochs_stage1:02d}] Train: {tr_l:.4f} ({tr_a*100:.2f}%) | Val: {va_l:.4f} ({va_a*100:.2f}%)")
        if va_l < best_val_loss:
            best_val_loss, best_val_acc = va_l, va_a
            torch.save(model.state_dict(), best_ckpt)

    # Giai đoạn 2: Mở khóa vài khối cuối + Early Stopping
    print(f"\n--- GIAI ĐOẠN 2 ({args.epochs_stage2} epochs) ---")
    unfreeze_last_blocks(model, num_blocks=args.unfreeze_blocks)
    opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_stage2, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=2)

    wait = 0
    for ep in range(1, args.epochs_stage2 + 1):
        tr_l, tr_a = run_epoch(model, train_loader, criterion, opt2, device)
        va_l, va_a = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(va_l)
        print(f"[GĐ2 {ep:02d}/{args.epochs_stage2:02d}] Train: {tr_l:.4f} ({tr_a*100:.2f}%) | Val: {va_l:.4f} ({va_a*100:.2f}%)")

        if va_l < best_val_loss:
            best_val_loss, best_val_acc, wait = va_l, va_a, 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> Lưu checkpoint tốt nhất (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Dừng sớm (Early Stopping) tại epoch {ep}.")
                break

    log_experiment(args.log_csv, [
        datetime.now().isoformat(timespec="seconds"),
        args.data_version, args.backbone, args.batch_size,
        args.lr_stage1, args.lr_stage2, args.label_smoothing,
        f"{best_val_loss:.4f}", f"{best_val_acc:.4f}", best_ckpt
    ])
    print(f"\nHoàn tất! Checkpoint lưu tại: {best_ckpt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data_v1")
    p.add_argument("--data_version", type=str, default="data_v1")
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_stage1", type=int, default=5)
    p.add_argument("--epochs_stage2", type=int, default=15)
    p.add_argument("--lr_stage1", type=float, default=1e-3)
    p.add_argument("--lr_stage2", type=float, default=1e-4)
    p.add_argument("--unfreeze_blocks", type=int, default=3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--log_csv", type=str, default="logs/experiments_log.csv")
    train_pipeline(p.parse_args())