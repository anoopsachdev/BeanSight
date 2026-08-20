#!/usr/bin/env python3
"""
Two-phase training pipeline for coffee bean classifiers.

Phase 1 — Feature Extraction: Frozen backbone, train head only (10 epochs, LR=1e-3)
Phase 2 — Fine-Tuning: Unfreeze last 2 blocks (20 epochs, LR=1e-4)

Supports training both roast and defect models independently.
Exports trained models to ONNX at completion.

Usage:
  python model/train.py --task roast  --data-dir data/roast
  python model/train.py --task defect --data-dir data/defect
  python model/train.py --task both
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from model.dataset import get_dataloaders
from model.network import (
    DEFECT_CLASSES,
    ROAST_CLASSES,
    CoffeeBeanClassifier,
)

# ── Constants ───────────────────────────────────────────────────────────

TASK_CONFIG = {
    "roast": {
        "classes": ROAST_CLASSES,
        "data_dir": "data/roast",
        "checkpoint": "checkpoints/roast_best.pt",
        "onnx_path": "checkpoints/roast_model.onnx",
    },
    "defect": {
        "classes": DEFECT_CLASSES,
        "data_dir": "data/defect",
        "checkpoint": "checkpoints/defect_best.pt",
        "onnx_path": "checkpoints/defect_model.onnx",
    },
}


class EarlyStopping:
    """Early stopping based on validation loss with patience."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def generate_report(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
) -> None:
    """Generate classification report and confusion matrix on test set."""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    print("\nCONFUSION MATRIX")
    print("-" * 40)
    cm = confusion_matrix(
        all_labels, all_preds,
        labels=range(len(class_names))
    )
    # Print with class labels
    header = "          " + " ".join(f"{c[:6]:>6}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        label = f"{class_names[i][:8]:<8}"
        values = " ".join(f"{v:>6}" for v in row)
        print(f"  {label}  {values}")


def train_model(
    task: str,
    data_dir: str | None = None,
    phase1_epochs: int = 10,
    phase2_epochs: int = 20,
    batch_size: int = 32,
    patience: int = 7,
) -> None:
    """
    Full training pipeline for a single task (roast or defect).

    Two-phase approach:
      Phase 1: Frozen backbone, LR=1e-3
      Phase 2: Unfreeze last 2 blocks, LR=1e-4
    """
    cfg = TASK_CONFIG[task]
    class_names = cfg["classes"]
    data_dir = data_dir or cfg["data_dir"]
    num_classes = len(class_names)

    print(f"\n{'='*60}")
    print(f"  TRAINING: {task.upper()} CLASSIFIER ({num_classes} classes)")
    print(f"{'='*60}\n")

    device = get_device()

    # Create data loaders
    print(f"📂 Loading data from {data_dir}")
    loaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        class_names=class_names,
    )

    if "train" not in loaders:
        raise FileNotFoundError(f"Training data not found at {data_dir}/train/")

    # Initialize model
    print(f"\n🏗️  Building EfficientNetB0 model ({num_classes} classes)")
    model = CoffeeBeanClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_time = time.time()

    # ── Phase 1: Feature Extraction ─────────────────────────────────
    print(f"\n🔒 Phase 1: Feature Extraction ({phase1_epochs} epochs, LR=1e-3)")
    print("   Backbone frozen, training head only\n")

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1,
    )
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:>2}/{phase1_epochs} │ "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} │ "
            f"LR: {lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"       ↳ 💾 Saved best model (val_loss={val_loss:.4f})")

        if early_stop(val_loss):
            print(f"  ⏹  Early stopping at epoch {epoch}")
            break

    # ── Phase 2: Fine-Tuning ────────────────────────────────────────
    print(f"\n🔓 Phase 2: Fine-Tuning ({phase2_epochs} epochs, LR=1e-4)")
    print("   Unfreezing last 2 backbone blocks\n")

    # Reload best checkpoint from phase 1
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.unfreeze_backbone(num_blocks=2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1,
    )
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(1, phase2_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:>2}/{phase2_epochs} │ "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} │ "
            f"LR: {lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"       ↳ 💾 Saved best model (val_loss={val_loss:.4f})")

        if early_stop(val_loss):
            print(f"  ⏹  Early stopping at epoch {epoch}")
            break

    # ── Final Evaluation & Export ───────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n⏱  Total training time: {elapsed / 60:.1f} minutes")

    # Reload best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Test set evaluation
    if "test" in loaders:
        test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
        print(f"\n📊 Test Results: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")
        generate_report(model, loaders["test"], class_names, device)

    # Export to ONNX
    print(f"\n📦 Exporting to ONNX ...")
    model.cpu()
    onnx_path = model.export_to_onnx(cfg["onnx_path"])
    print(f"  ✅ ONNX model saved: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Train coffee bean classifiers")
    parser.add_argument(
        "--task",
        choices=["roast", "defect", "both"],
        default="both",
        help="Which model to train",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    tasks = ["roast", "defect"] if args.task == "both" else [args.task]

    for task in tasks:
        train_model(
            task=task,
            data_dir=args.data_dir,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
