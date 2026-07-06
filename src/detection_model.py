"""
detection_model.py
===================
Stage 1 of the pipeline: locate every product on the shelf.

Two interchangeable backends are provided (select via config.DETECTION_BACKEND):

    "retinanet_pretrained" (DEFAULT)
        Loads the ready-trained SKU110K RetinaNet checkpoint
        (models/iou_resnet50_csv_06.h5) - the official SKU110K architecture:
        ResNet50 backbone + FPN + classification/regression/IoU-Net heads.
        No training required - this is the checkpoint the user already has.

    "yolov8"
        Trains/loads a YOLOv8 model from scratch (see train_detection.py).
        Kept as an alternative if you ever want to retrain instead of using
        the provided checkpoint.

Both backends expose the same `detect()` / `estimate_shelf_gaps()` interface,
so the rest of the pipeline (classification_model.py, pipeline.py, app.py)
does not need to know which one is active.
"""

from pathlib import Path
from typing import List, Dict

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# ============================================================
# Backend 1 (DEFAULT): Pretrained SKU110K RetinaNet checkpoint
# ============================================================
class RetinaNetShelfDetector:
    """
    Loads and runs inference with the pretrained SKU110K RetinaNet checkpoint
    (iou_resnet50_csv_06.h5), reconstructing the inference-time graph
    (anchors -> box regression -> clipping -> score-threshold + NMS) on top
    of the raw regression/classification/iou heads baked into the checkpoint.
    """

    def __init__(self, weights_path: str = None):
        from tensorflow import keras
        from src.keras_retinanet_layers import CUSTOM_OBJECTS

        weights_path = weights_path or str(config.RETINANET_WEIGHTS)
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"RetinaNet checkpoint not found at {weights_path}. "
                f"Place iou_resnet50_csv_06.h5 under {config.MODELS_DIR}/"
            )

        print(f"[INFO] Loading pretrained SKU110K RetinaNet checkpoint: {weights_path}")
        self.training_model = keras.models.load_model(
            weights_path, custom_objects=CUSTOM_OBJECTS, compile=False
        )

        self.inference_model = self._build_inference_model()

    def _build_inference_model(self):
        from tensorflow import keras
        from src.keras_retinanet_layers import Anchors, RegressBoxes, ClipBoxes, FilterDetections

        image_input = self.training_model.inputs[0]
        regression, classification, iou = self.training_model.outputs

        pyramid_layer_names = [f"P{lvl}" for lvl in config.RETINANET_PYRAMID_LEVELS]
        pyramid_features = [self.training_model.get_layer(name).output for name in pyramid_layer_names]

        anchors_per_level = []
        for feature, size, stride in zip(
            pyramid_features, config.RETINANET_ANCHOR_SIZES, config.RETINANET_ANCHOR_STRIDES
        ):
            anchors_per_level.append(
                Anchors(
                    size=size, stride=stride,
                    ratios=config.RETINANET_ANCHOR_RATIOS, scales=config.RETINANET_ANCHOR_SCALES,
                )(feature)
            )
        anchors = keras.layers.Concatenate(axis=1)(anchors_per_level) if len(anchors_per_level) > 1 else anchors_per_level[0]

        boxes = RegressBoxes()([anchors, regression])
        boxes = ClipBoxes()([image_input, boxes])

        filter_inputs = [boxes, classification, iou] if config.RETINANET_USE_IOU_HEAD else [boxes, classification]
        detections_boxes, detections_scores = FilterDetections(
            score_threshold=config.RETINANET_SCORE_THRESHOLD,
            nms_threshold=config.RETINANET_NMS_IOU_THRESHOLD,
            max_detections=config.RETINANET_MAX_DETECTIONS,
        )(filter_inputs)

        return keras.Model(inputs=image_input, outputs=[detections_boxes, detections_scores])

    @staticmethod
    def _preprocess(image_path: str):
        import cv2

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        h, w = image.shape[:2]
        min_side, max_side = config.RETINANET_MIN_IMAGE_SIDE, config.RETINANET_MAX_IMAGE_SIDE
        scale = min_side / min(h, w)
        if max(h, w) * scale > max_side:
            scale = max_side / max(h, w)

        resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))

        mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        resized = resized[..., ::-1] - mean

        return np.expand_dims(resized, axis=0), scale, (h, w)

    def detect(self, image_path: str) -> List[Dict]:
        batch, scale, (orig_h, orig_w) = self._preprocess(image_path)
        boxes, scores = self.inference_model.predict(batch, verbose=0)

        boxes, scores = boxes[0], scores[0]
        valid = scores > 0

        detections = []
        for box, score in zip(boxes[valid], scores[valid]):
            x1, y1, x2, y2 = (box / scale).tolist()
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, orig_w), min(y2, orig_h)
            detections.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(float(score), 3),
            })
        return detections

    def estimate_shelf_gaps(self, image_path: str, image_width: int, image_height: int,
                             gap_threshold_ratio: float = 0.08) -> List[Dict]:
        detections = self.detect(image_path)
        if not detections:
            return []

        boxes = sorted(detections, key=lambda d: d["bbox"][0])
        gaps = []
        for i in range(len(boxes) - 1):
            right_edge_of_current = boxes[i]["bbox"][2]
            left_edge_of_next = boxes[i + 1]["bbox"][0]
            gap_width = left_edge_of_next - right_edge_of_current
            if gap_width > gap_threshold_ratio * image_width:
                gaps.append({
                    "gap_start_x": right_edge_of_current,
                    "gap_end_x": left_edge_of_next,
                    "gap_width_px": round(gap_width, 1),
                })
        return gaps


# ============================================================
# Backend 2 (alternative): YOLOv8 trained from scratch
# ============================================================
class ShelfProductDetector:
    """Thin wrapper around a YOLOv8 model trained on SKU110K (see train_detection.py)."""

    def __init__(self, weights_path: str = None):
        from ultralytics import YOLO
        weights_path = weights_path or str(config.DETECTION_WEIGHTS)
        if weights_path and Path(weights_path).exists():
            self.model = YOLO(weights_path)
        else:
            self.model = YOLO(config.DETECTION_MODEL_ARCH)

    @staticmethod
    def train(data_yaml: str = None, base_model: str = None, epochs: int = None,
              img_size: int = None, batch_size: int = None):
        from ultralytics import YOLO

        data_yaml = data_yaml or str(config.YOLO_DATA_DIR / "sku110k.yaml")
        base_model = base_model or config.DETECTION_MODEL_ARCH
        epochs = epochs or config.DETECTION_EPOCHS
        img_size = img_size or config.DETECTION_IMG_SIZE
        batch_size = batch_size or config.DETECTION_BATCH_SIZE

        model = YOLO(base_model)
        results = model.train(
            data=data_yaml, epochs=epochs, imgsz=img_size, batch=batch_size,
            project=str(config.MODELS_DIR), name="shelf_detector_run",
            patience=10, optimizer="AdamW", lr0=1e-3, cos_lr=True, augment=True, verbose=True,
        )
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if best_weights.exists():
            best_weights.replace(config.DETECTION_WEIGHTS)
            print(f"[OK] Best detector weights saved to {config.DETECTION_WEIGHTS}")
        return results

    @staticmethod
    def evaluate(data_yaml: str = None):
        from ultralytics import YOLO
        data_yaml = data_yaml or str(config.YOLO_DATA_DIR / "sku110k.yaml")
        model = YOLO(str(config.DETECTION_WEIGHTS))
        metrics = model.val(data=data_yaml)
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        return metrics

    def detect(self, image_path: str) -> List[Dict]:
        results = self.model.predict(
            source=image_path, conf=config.DETECTION_CONF_THRESHOLD,
            iou=config.DETECTION_IOU_THRESHOLD, verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(conf, 3),
                })
        return detections

    def estimate_shelf_gaps(self, image_path: str, image_width: int, image_height: int,
                             gap_threshold_ratio: float = 0.08) -> List[Dict]:
        detections = self.detect(image_path)
        if not detections:
            return []
        boxes = sorted(detections, key=lambda d: d["bbox"][0])
        gaps = []
        for i in range(len(boxes) - 1):
            right_edge_of_current = boxes[i]["bbox"][2]
            left_edge_of_next = boxes[i + 1]["bbox"][0]
            gap_width = left_edge_of_next - right_edge_of_current
            if gap_width > gap_threshold_ratio * image_width:
                gaps.append({
                    "gap_start_x": right_edge_of_current,
                    "gap_end_x": left_edge_of_next,
                    "gap_width_px": round(gap_width, 1),
                })
        return gaps


# ============================================================
# Factory - the rest of the codebase calls this, never the classes directly
# ============================================================
def build_shelf_detector():
    """
    Returns the configured detection backend (config.DETECTION_BACKEND).
    This is what pipeline.py imports and calls - swapping backends is a
    one-line config change, no other code needs to be touched.
    """
    if config.DETECTION_BACKEND == "retinanet_pretrained":
        try:
            return RetinaNetShelfDetector()
        except (ImportError, FileNotFoundError) as e:
            print(f"[WARN] RetinaNet backend unavailable: {e}")
            print("[INFO] Falling back to YOLOv8 backend.")
            return ShelfProductDetector()
    elif config.DETECTION_BACKEND == "yolov8":
        return ShelfProductDetector()
    else:
        raise ValueError(f"Unknown DETECTION_BACKEND: {config.DETECTION_BACKEND}")


if __name__ == "__main__":
    detector = build_shelf_detector()
    sample_results = detector.detect("data/raw/SKU110K/images/sample_shelf.jpg")
    print(f"Detected {len(sample_results)} products using backend '{config.DETECTION_BACKEND}'")
