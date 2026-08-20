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


class YOLOv8ONNXPredictor:
    """
    Zero-PyTorch ONNX Runtime predictor for YOLOv8 object detection.

    Implements:
      1. Letterbox aspect-ratio preserving resize to 640x640 with border padding.
      2. ONNX Runtime CPU inference.
      3. Pure NumPy Non-Maximum Suppression (NMS).
      4. PIL ImageDraw annotation producing Base64 visual output.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        if ort is None:
            raise ImportError("onnxruntime is required. pip install onnxruntime")

        self.model_path = Path(model_path)
        self.class_names = class_names or ["defect", "bean"]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found: {self.model_path}")

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
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.img_size = self.input_shape[2] if len(self.input_shape) >= 4 else 640

    def is_loaded(self) -> bool:
        return self.session is not None

    def letterbox(self, image: Image.Image, target_size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize image with aspect ratio preservation and border padding."""
        orig_w, orig_h = image.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.LANCZOS)
        padded = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        padded.paste(resized, (pad_x, pad_y))

        img_array = np.array(padded, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

        return img_array, scale, (pad_x, pad_y)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        """Pure NumPy Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def predict(self, image: Image.Image) -> dict:
        """
        Run YOLOv8 object detection on roasted beans.

        Returns structured detections and base64 annotated image.
        """
        orig_w, orig_h = image.size
        input_tensor, scale, (pad_x, pad_y) = self.letterbox(image, self.img_size)

        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]  # Shape: (1, 4+num_classes, 8400)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # YOLOv8 output transpose: (1, 8400, 4+num_classes)
        preds = np.transpose(outputs[0], (1, 0))
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence threshold
        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return {
                "defect_count": 0,
                "defect_summary": {},
                "detections": [],
                "inference_time_ms": round(elapsed_ms, 2),
                "annotated_image": None,
            }

        # Convert xywh to xyxy on original image coordinates
        x_c, y_c, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (x_c - w / 2 - pad_x) / scale
        y1 = (y_c - h / 2 - pad_y) / scale
        x2 = (x_c + w / 2 - pad_x) / scale
        y2 = (y_c + h / 2 - pad_y) / scale

        # Clip to image boundaries
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes_xyxy = np.column_stack([x1, y1, x2, y2])

        # Apply NMS
        keep_indices = self._nms(boxes_xyxy, confidences, self.iou_threshold)
        final_boxes = boxes_xyxy[keep_indices]
        final_confs = confidences[keep_indices]
        final_classes = class_ids[keep_indices]

        # Structure detections
        detections = []
        summary = {}
        for box, conf, cls_id in zip(final_boxes, final_confs, final_classes):
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"defect_{cls_id}"
            detections.append({
                "class": cls_name,
                "confidence": round(float(conf), 4),
                "box": [round(float(coord), 1) for coord in box],
            })
            summary[cls_name] = summary.get(cls_name, 0) + 1

        # Generate base64 annotated preview
        annotated_b64 = self.annotate_image(image, detections)

        return {
            "defect_count": len(detections),
            "defect_summary": summary,
            "detections": detections,
            "inference_time_ms": round(elapsed_ms, 2),
            "annotated_image": annotated_b64,
        }

    def annotate_image(self, image: Image.Image, detections: list[dict]) -> str | None:
        """Draw bounding boxes and labels onto the image, returning base64 JPEG."""
        import base64
        import io
        from PIL import ImageDraw

        if not detections:
            return None

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        for det in detections:
            box = det["box"]
            cls_name = det["class"]
            conf = det["confidence"]

            # Draw outer box
            draw.rectangle(box, outline="#e5a968", width=3)

            # Label banner
            x1, y1, _, _ = box
            label_text = f"{cls_name} {int(conf * 100)}%"
            draw.rectangle([x1, max(0, y1 - 18), x1 + len(label_text) * 8 + 6, max(0, y1)], fill="#e5a968")
            draw.text((x1 + 3, max(0, y1 - 16)), label_text, fill="#120d09")

        buffer = io.BytesIO()
        img_draw.save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def __repr__(self) -> str:
        return f"YOLOv8ONNXPredictor(model={self.model_path.name})"

