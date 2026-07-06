"""
train_classification.py
=========================
Convenience entry point for training the Stage-2 CNN product-category
classifier (ResNet50 by default).

Expects data laid out as:
    data/processed/crops/train_labeled/<category>/*.jpg
    data/processed/crops/val_labeled/<category>/*.jpg

Usage:
    python train_classification.py --backbone resnet50 --epochs 25
"""

import argparse
import config
from src.classification_model import train_classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=str(config.CROPS_DIR / "train_labeled"))
    parser.add_argument("--val_dir", default=str(config.CROPS_DIR / "val_labeled"))
    parser.add_argument("--backbone", default=config.CNN_BACKBONE,
                         choices=["resnet50", "efficientnet_b0", "mobilenet_v3_large", "vgg16"])
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    args = parser.parse_args()

    model, best_acc = train_classifier(
        args.train_dir, args.val_dir, backbone_name=args.backbone, epochs=args.epochs
    )
    print(f"\n[DONE] Best validation accuracy: {best_acc:.4f}")
