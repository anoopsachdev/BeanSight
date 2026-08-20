"""
YOLOv8 training and ONNX export pipeline for Roasted Coffee Bean Defects.

Trains YOLOv8-nano on data/roasted_defects/data.yaml, evaluates performance,
and automatically exports the final model to ONNX format at `checkpoints/defect_yolo.onnx`.

Usage:
  python model/train_yolo.py --data data/roasted_defects/data.yaml --epochs 30 --imgsz 640
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def train_and_export_yolo(
    data_yaml: str = "data/roasted_defects/data.yaml",
    epochs: int = 30,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cpu",
    output_onnx: str = "checkpoints/defect_yolo.onnx",
) -> Path:
    """Train YOLOv8n and export directly to ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Error: `ultralytics` package not installed.")
        print("Run: pip install -r requirements-train.txt")
        sys.exit(1)

    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Error: data.yaml not found at {data_path}")
        print("Run: python data/download_roboflow.py first.")
        sys.exit(1)

    print(f"🚀 Initializing YOLOv8-nano pretrained baseline...")
    model = YOLO("yolov8n.pt")

    print(f"🏋️ Training YOLOv8n on {data_yaml} ({epochs} epochs, imgsz={imgsz})...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project="runs/detect",
        name="roasted_defect_train",
        save=True,
    )

    print(f"📦 Exporting best weights to ONNX format (imgsz={imgsz}, simplify=True)...")
    exported_onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
    )

    out_path = Path(output_onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported_onnx_path, out_path)
    print(f"✅ ONNX model successfully saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Roasted Bean Defects")
    parser.add_argument("--data", type=str, default="data/roasted_defects/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--output", type=str, default="checkpoints/defect_yolo.onnx", help="Target ONNX checkpoint")

    args = parser.parse_args()
    train_and_export_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_onnx=args.output,
    )
