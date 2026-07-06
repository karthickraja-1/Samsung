"""
check_retinanet_checkpoint.py
================================
Standalone sanity check for the pretrained SKU110K RetinaNet checkpoint.

Run this FIRST after installing requirements.txt, before starting the full
Streamlit app, to confirm:
    1. TensorFlow/Keras can deserialize the checkpoint's custom layers.
    2. The inference graph (anchors -> box decode -> NMS) builds without error.
    3. A prediction can be produced on a real image.

Usage:
    python check_retinanet_checkpoint.py path/to/shelf_photo.jpg
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.detection_model import RetinaNetShelfDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_retinanet_checkpoint.py path/to/shelf_photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"Checkpoint path : {config.RETINANET_WEIGHTS}")
    print(f"Checkpoint size : {Path(config.RETINANET_WEIGHTS).stat().st_size / 1e6:.1f} MB")
    print("Loading model (this rebuilds the inference graph on top of the checkpoint) ...")

    detector = RetinaNetShelfDetector()
    print("[OK] Model loaded and inference graph built successfully.")

    print(f"Running detection on: {image_path}")
    detections = detector.detect(image_path)
    print(f"[OK] Detected {len(detections)} products.")

    for d in detections[:10]:
        print(f"   bbox={d['bbox']}  confidence={d['confidence']}")
    if len(detections) > 10:
        print(f"   ... and {len(detections) - 10} more")


if __name__ == "__main__":
    main()
