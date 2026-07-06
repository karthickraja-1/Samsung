"""
classification_model.py
========================
Stage 2 of the pipeline: classify each cropped product image (produced by
detection_model.py / data_preprocessing.py) into a product category.

Uses transfer learning on a torchvision CNN backbone. ResNet50 is the default
(config.CNN_BACKBONE) but EfficientNet-B0, MobileNetV3-Large, and VGG16 are
also supported - swap the backbone in config.py without touching this file.

Training strategy:
    1. Load an ImageNet-pretrained backbone, freeze it.
    2. Replace the final classifier head with a new fully-connected layer sized
       to the number of product categories (config.DEFAULT_CLASSES).
    3. Train only the head for `FREEZE_BACKBONE_EPOCHS` epochs (fast convergence).
    4. Unfreeze the backbone and fine-tune end-to-end at a lower learning rate.
"""

import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Backbone factory
# ------------------------------------------------------------------
def build_backbone(name: str, num_classes: int) -> nn.Module:
    """Return an ImageNet-pretrained CNN with its classifier head replaced."""
    name = name.lower()

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_params = model.fc.parameters()

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_params = model.classifier[1].parameters()

    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        head_params = model.classifier[3].parameters()

    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        head_params = model.classifier[6].parameters()

    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return model, head_params


def freeze_backbone(model: nn.Module, backbone_name: str) -> None:
    """Freeze every parameter except the newly-added classifier head."""
    head_module_names = {
        "resnet50": "fc",
        "efficientnet_b0": "classifier",
        "mobilenet_v3_large": "classifier",
        "vgg16": "classifier",
    }
    head_name = head_module_names[backbone_name.lower()]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head_name)


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ------------------------------------------------------------------
# Data transforms
# ------------------------------------------------------------------
def get_transforms(img_size: int = config.CNN_IMG_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train_classifier(
    train_dir: str,
    val_dir: str,
    backbone_name: str = config.CNN_BACKBONE,
    epochs: int = config.CNN_EPOCHS,
    batch_size: int = config.CNN_BATCH_SIZE,
    lr: float = config.CNN_LEARNING_RATE,
    freeze_epochs: int = config.FREEZE_BACKBONE_EPOCHS,
):
    """
    Trains the product-category classifier.

    Expects `train_dir` / `val_dir` in torchvision ImageFolder layout:
        train_dir/beverages/*.jpg
        train_dir/snacks/*.jpg
        ...
    (i.e. crops from data_preprocessing.py organised into per-class folders
    after merging with a labeled product-category dataset.)
    """
    train_tf, eval_tf = get_transforms()
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    class_names = train_ds.classes
    with open(config.CLASS_NAMES_FILE, "w") as f:
        json.dump(class_names, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=config.CNN_NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=config.CNN_NUM_WORKERS)

    model, head_params = build_backbone(backbone_name, num_classes=len(class_names))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: train classifier head only (backbone frozen)
    freeze_backbone(model, backbone_name)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    for epoch in range(epochs):
        if epoch == freeze_epochs:
            print("[INFO] Unfreezing backbone for fine-tuning ...")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr / 10)

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": backbone_name,
                "class_names": class_names,
            }, config.CLASSIFIER_WEIGHTS)
            print(f"[OK] New best model saved (val_acc={val_acc:.4f}) -> {config.CLASSIFIER_WEIGHTS}")

    return model, best_val_acc


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ------------------------------------------------------------------
# Inference wrapper used by pipeline.py / app.py
# ------------------------------------------------------------------
class ProductClassifier:
    def __init__(self, weights_path: str = None):
        weights_path = weights_path or str(config.CLASSIFIER_WEIGHTS)
        if not Path(weights_path).exists():
            print(
                f"[WARN] Classifier weights not found at {weights_path}. "
                "Using a lightweight fallback classifier."
            )
            self.model = None
            self.class_names = config.DEFAULT_CLASSES
            self.backbone_name = config.CNN_BACKBONE
            _, self.transform = get_transforms()
            return

        checkpoint = torch.load(weights_path, map_location=DEVICE)

        self.class_names: List[str] = checkpoint["class_names"]
        self.backbone_name: str = checkpoint["backbone"]

        self.model, _ = build_backbone(self.backbone_name, num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        _, self.transform = get_transforms()

    @torch.no_grad()
    def predict(self, pil_image) -> dict:
        """Returns {"label": str, "confidence": float, "top3": [(label, prob), ...]}"""
        tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
        if self.model is None:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "top3": [("unknown", 0.0)],
            }
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        top3_probs, top3_idx = torch.topk(probs, k=min(3, len(self.class_names)))
        top3 = [(self.class_names[i], round(float(p), 4)) for p, i in zip(top3_probs, top3_idx)]

        return {
            "label": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the product-category CNN classifier")
    parser.add_argument("--train_dir", default=str(config.CROPS_DIR / "train_labeled"))
    parser.add_argument("--val_dir", default=str(config.CROPS_DIR / "val_labeled"))
    parser.add_argument("--backbone", default=config.CNN_BACKBONE)
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    args = parser.parse_args()

    train_classifier(args.train_dir, args.val_dir, backbone_name=args.backbone, epochs=args.epochs)
