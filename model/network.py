"""
Dual EfficientNetB0 model architecture for coffee bean analysis.

Two independent models:
  1. Roast classifier  — 4 classes (Dark, Green, Light, Medium)
  2. Defect detector   — 17 classes (Broken, Cut, Dry Cherry, ...)

Each model uses EfficientNetB0 backbone with a custom classification head:
  GlobalAvgPool → Dropout(0.3) → Linear(1280, num_classes)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

try:
    import onnx
except ImportError:
    onnx = None

# ── Class definitions ───────────────────────────────────────────────────

ROAST_CLASSES = ["Dark", "Green", "Light", "Medium"]

DEFECT_CLASSES = [
    "Broken",
    "Cut",
    "Dry Cherry",
    "Fade",
    "Floater",
    "Full Black",
    "Full Sour",
    "Fungus Damage",
    "Husk",
    "Immature",
    "Parchment",
    "Partial Black",
    "Partial Sour",
    "Severe Insect Damage",
    "Shell",
    "Slight Insect Damage",
    "Withered",
]


class CoffeeBeanClassifier(nn.Module):
    """
    EfficientNetB0 + custom classification head.

    Architecture:
      - Backbone: EfficientNetB0 (pretrained on ImageNet via timm)
      - Head: AdaptiveAvgPool2d → Dropout(0.3) → Linear(1280, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for training. pip install timm")

        self.num_classes = num_classes
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,           # Remove original classifier
            global_pool="",          # We'll add our own pooling
        )
        self.feature_dim = 1280  # EfficientNetB0 output channels

        # Custom classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        # Freeze backbone by default for phase-1 training
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (feature extraction mode)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_blocks: int = 2) -> None:
        """
        Unfreeze the last N blocks of the EfficientNet backbone.
        EfficientNetB0 has 7 blocks (0–6).
        """
        # First, freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks
        blocks = list(self.backbone.blocks.children())
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze batch norm in unfrozen blocks
        unfrozen = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        total = sum(1 for _ in self.backbone.parameters())
        print(f"  Unfroze {unfrozen}/{total} backbone parameters ({num_blocks} blocks)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def export_to_onnx(
        self,
        save_path: str | Path,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        validate: bool = True,
    ) -> Path:
        """
        Export the model to ONNX format.

        Args:
            save_path: Where to save the .onnx file.
            input_shape: Example input shape for tracing.
            validate: Whether to validate the exported model.

        Returns:
            Path to the saved ONNX file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        original_device = next(self.parameters()).device

        # Always export from CPU so ONNX Runtime (CPU-only) and PyTorch
        # use the same execution path, eliminating MPS vs CPU divergence.
        self.cpu()
        dummy_input = torch.randn(*input_shape, device="cpu")

        print(f"  Exporting to ONNX: {save_path}")
        torch.onnx.export(
            self,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        if validate:
            self._validate_onnx(save_path, dummy_input)

        # Restore model to original device
        self.to(original_device)

        print(f"  ✅ ONNX export complete: {save_path}")
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"     File size: {file_size_mb:.1f} MB")
        return save_path

    def _validate_onnx(self, onnx_path: Path, dummy_input: torch.Tensor) -> None:
        """Validate ONNX model structure and output consistency."""
        if onnx is None:
            print("  ⚠️  onnx package not installed, skipping validation")
            return

        # Structural validation
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("  ✓ ONNX model structure is valid")

        # Output consistency check
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            ort_input = {
                "input": dummy_input.cpu().numpy(),
            }
            ort_output = session.run(None, ort_input)[0]

            # Compare with PyTorch output (both on CPU, no MPS divergence)
            with torch.no_grad():
                pt_output = self(dummy_input).numpy()

            max_diff = np.max(np.abs(pt_output - ort_output))
            print(f"  ✓ Max output difference (PT vs ONNX): {max_diff:.6f}")
            assert max_diff < 5e-3, f"Output mismatch too large: {max_diff}"
        except ImportError:
            print("  ⚠️  onnxruntime not installed, skipping output validation")
