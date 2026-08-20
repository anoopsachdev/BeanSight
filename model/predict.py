"""
ONNX Runtime inference for coffee bean classification.

This module has ZERO PyTorch dependency — it uses only:
  - onnxruntime for model inference
  - Pillow for image loading
  - NumPy for preprocessing

This is what runs in the production Docker container.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ── ImageNet normalization stats ────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(
    image: Image.Image,
    img_size: int = 224,
) -> np.ndarray:
    """
    Preprocess an image for ONNX inference (no PyTorch required).

    Steps:
      1. Resize to img_size × img_size
      2. Convert to float32 [0, 1]
      3. Normalize with ImageNet mean/std
      4. Transpose to CHW format
      5. Add batch dimension

    Returns:
        np.ndarray of shape (1, 3, img_size, img_size)
    """
    # Resize
    image = image.convert("RGB")
    image = image.resize((img_size, img_size), Image.LANCZOS)

    # To numpy float32 [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize (per-channel)
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW
    img_array = img_array.transpose(2, 0, 1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class ONNXPredictor:
    """
    ONNX Runtime predictor for coffee bean images.

    Loads an ONNX model once and provides fast CPU-based inference.
    """

    def __init__(self, model_path: str | Path, class_names: list[str]):
        if ort is None:
            raise ImportError("onnxruntime is required. pip install onnxruntime")

        self.model_path = Path(model_path)
        self.class_names = class_names

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Configure session for optimal CPU performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single image.

        Args:
            image: PIL Image (any size, will be resized).

        Returns:
            {
                "prediction": "Medium",
                "confidence": 0.94,
                "probabilities": {"Dark": 0.02, "Green": 0.01, ...},
                "inference_time_ms": 45.2,
            }
        """
        # Preprocess
        input_array = preprocess_image(image)

        # Run inference
        start = time.perf_counter()
        raw_output = self.session.run(
            [self.output_name],
            {self.input_name: input_array},
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Apply softmax to get probabilities
        probs = softmax(raw_output)[0]

        # Build result
        prob_dict = {
            name: round(float(prob), 4)
            for name, prob in zip(self.class_names, probs)
        }
        top_idx = int(np.argmax(probs))

        return {
            "prediction": self.class_names[top_idx],
            "confidence": round(float(probs[top_idx]), 4),
            "probabilities": prob_dict,
            "inference_time_ms": round(elapsed_ms, 2),
        }

    def is_loaded(self) -> bool:
        """Check if the ONNX session is properly loaded."""
        return self.session is not None

    def __repr__(self) -> str:
        return (
            f"ONNXPredictor(model={self.model_path.name}, "
            f"classes={len(self.class_names)})"
        )
