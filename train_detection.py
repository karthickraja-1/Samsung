"""
train_detection.py
====================
Convenience entry point for training the Stage-1 YOLOv8 shelf-product
detector on SKU110K.

Steps this script assumes have already been run once:
    python src/data_preprocessing.py --split all

Usage:
    python train_detection.py --epochs 50 --batch 16
"""

import argparse
from src.detection_model import ShelfProductDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    ShelfProductDetector.train(epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz)
    print("\n[DONE] Now run: python -c \"from src.detection_model import ShelfProductDetector as D; D.evaluate()\"")
