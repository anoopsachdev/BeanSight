"""
Unit tests for the YOLOv8 post-processing / summarisation logic.

These construct a predictor without loading an ONNX session so they run
without model files or onnxruntime.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from model.predict import YOLOv8ONNXPredictor

CLASSES = ["Broken", "Quaker", "Normal", "Scorched"]


def make_predictor() -> YOLOv8ONNXPredictor:
    p = YOLOv8ONNXPredictor.__new__(YOLOv8ONNXPredictor)
    p.class_names = CLASSES
    p.conf_threshold = 0.25
    p.iou_threshold = 0.45
    p.normal_class = "Normal"
    p.img_size = 640
    return p


def fake_output(rows: list[tuple[float, float, float, float, int, float]], num_anchors: int = 8400) -> np.ndarray:
    """Build a (1, 4+num_classes, num_anchors) tensor with the given (cx, cy, w, h, cls, score) boxes."""
    out = np.zeros((1, 4 + len(CLASSES), num_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out


class TestSummarize:
    def test_normal_beans_are_not_defects(self):
        p = make_predictor()
        detections = [
            {"class": "Normal", "confidence": 0.9, "box": [0, 0, 10, 10]},
            {"class": "Normal", "confidence": 0.7, "box": [20, 20, 30, 30]},
            {"class": "Broken", "confidence": 0.6, "box": [40, 40, 50, 50]},
        ]
        s = p.summarize(detections)
        assert s["bean_count"] == 3
        assert s["normal_count"] == 2
        assert s["defect_count"] == 1
        assert s["defect_summary"] == {"Broken": 1}
        assert s["class_summary"] == {"Normal": 2, "Broken": 1}
        assert s["mean_defect_confidence"] == 0.6
        assert abs(s["mean_confidence"] - (0.9 + 0.7 + 0.6) / 3) < 1e-4

    def test_empty(self):
        s = make_predictor().summarize([])
        assert s["bean_count"] == 0
        assert s["defect_count"] == 0
        assert s["defect_summary"] == {}
        assert s["mean_confidence"] == 0.0


class TestPostprocess:
    def test_decodes_and_filters_by_confidence(self):
        p = make_predictor()
        out = fake_output([
            (320, 320, 100, 100, 2, 0.9),   # Normal, keep
            (100, 100, 40, 40, 0, 0.5),     # Broken, keep
            (500, 500, 40, 40, 3, 0.1),     # Scorched, below threshold
        ])
        dets = p._postprocess(out, orig_size=(640, 640), scale=1.0, pad=(0, 0))
        assert [d["class"] for d in sorted(dets, key=lambda d: d["box"][0])] == ["Broken", "Normal"]
        normal = next(d for d in dets if d["class"] == "Normal")
        assert normal["box"] == [270.0, 270.0, 370.0, 370.0]

    def test_nms_merges_overlapping_boxes(self):
        p = make_predictor()
        out = fake_output([
            (320, 320, 100, 100, 1, 0.9),
            (322, 322, 100, 100, 1, 0.8),   # heavy overlap with the first → suppressed
        ])
        dets = p._postprocess(out, orig_size=(640, 640), scale=1.0, pad=(0, 0))
        assert len(dets) == 1
        assert dets[0]["confidence"] == 0.9

    def test_letterbox_coordinates_are_mapped_back(self):
        p = make_predictor()
        # 320x640 image letterboxed into 640x640: scale=1, pad_x=160
        out = fake_output([(320, 320, 100, 100, 2, 0.9)])
        dets = p._postprocess(out, orig_size=(320, 640), scale=1.0, pad=(160, 0))
        assert dets[0]["box"] == [110.0, 270.0, 210.0, 370.0]


class TestAnnotate:
    def test_returns_data_url_and_handles_normal_only(self):
        p = make_predictor()
        img = Image.new("RGB", (200, 200), "white")
        dets = [
            {"class": "Normal", "confidence": 0.9, "box": [10, 10, 60, 60]},
            {"class": "Scorched", "confidence": 0.7, "box": [100, 5, 150, 60]},  # label near top edge
        ]
        data_url = p.annotate_image(img, dets)
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_no_detections_returns_none(self):
        assert make_predictor().annotate_image(Image.new("RGB", (50, 50)), []) is None
