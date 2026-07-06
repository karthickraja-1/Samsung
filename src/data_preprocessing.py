"""
data_preprocessing.py
======================
Prepares the SKU110K dataset (https://github.com/eg4000/SKU110K_CVPR19)
for the two-stage CNN pipeline used by the Smart Retail Shelf Assistant:

    Stage 1 (Detection)      : locate every product bounding box on a shelf image.
    Stage 2 (Classification) : crop each bounding box and classify its product category.

SKU110K ships as:
    images/                       -> shelf photographs
    annotations_train.csv         -> image_name, x1, y1, x2, y2, class, image_width, image_height
    annotations_val.csv
    annotations_test.csv

Because SKU110K only labels a single generic class ("object"), this script:
    1. Converts the CSV annotations into YOLO-format .txt files for detector training.
    2. Optionally crops each bounding box out of the shelf image so those crops can be
       fed into a second, separately-labeled product-category dataset/classifier
       (see classification_model.py). Category labels for the crops must come from a
       paired classification dataset (e.g. Freiburg Groceries, RPC) or manual labeling -
       SKU110K crops without external labels are used here only for weak/unsupervised
       pretraining or clustering, not supervised classification.

Usage:
    python src/data_preprocessing.py --split train
    python src/data_preprocessing.py --split all --crop
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_annotations(split: str) -> pd.DataFrame:
    """Load a SKU110K annotation CSV (no header) into a DataFrame."""
    filename = {
        "train": config.SKU110K_ANNOTATION_TRAIN,
        "val": config.SKU110K_ANNOTATION_VAL,
        "test": config.SKU110K_ANNOTATION_TEST,
    }[split]

    csv_path = config.RAW_DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {csv_path}\n"
            f"Download SKU110K from {config.SKU110K_URL} and place it under "
            f"{config.RAW_DATA_DIR}"
        )

    df = pd.read_csv(csv_path, header=None, names=config.SKU110K_COLUMNS)
    return df


def convert_to_yolo_format(split: str) -> None:
    """
    Convert SKU110K bounding boxes to YOLO txt format:
        <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

    Since SKU110K has a single object class, class_id is always 0 ("product").
    Produces one .txt file per image inside data/processed/yolo_format/<split>/labels/
    and copies (or symlinks) images into .../images/.
    """
    df = load_annotations(split)

    img_out_dir = config.YOLO_DATA_DIR / split / "images"
    lbl_out_dir = config.YOLO_DATA_DIR / split / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("image_name")

    for image_name, group in tqdm(grouped, desc=f"Converting {split} annotations to YOLO format"):
        src_img_path = config.RAW_DATA_DIR / config.SKU110K_IMAGE_SUBDIR / image_name
        if not src_img_path.exists():
            continue

        dst_img_path = img_out_dir / image_name
        if not dst_img_path.exists():
            try:
                os.symlink(src_img_path, dst_img_path)
            except OSError:
                shutil.copy(src_img_path, dst_img_path)

        label_lines = []
        for _, row in group.iterrows():
            img_w, img_h = row["image_width"], row["image_height"]
            x_center = ((row["x1"] + row["x2"]) / 2) / img_w
            y_center = ((row["y1"] + row["y2"]) / 2) / img_h
            width = (row["x2"] - row["x1"]) / img_w
            height = (row["y2"] - row["y1"]) / img_h
            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_path = lbl_out_dir / f"{Path(image_name).stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    # Write the YOLO dataset YAML used by ultralytics for training.
    yaml_path = config.YOLO_DATA_DIR / "sku110k.yaml"
    yaml_content = f"""\
path: {config.YOLO_DATA_DIR}
train: train/images
val: val/images
test: test/images

names:
  0: product
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[OK] YOLO-format labels written to {lbl_out_dir}")
    print(f"[OK] Dataset YAML written to {yaml_path}")


def crop_bounding_boxes(split: str, padding: int = 4) -> None:
    """
    Crop every annotated product bounding box out of its shelf image and save it
    to data/processed/crops/<split>/. These crops are the input to the Stage-2
    CNN classifier once paired with product-category labels.
    """
    df = load_annotations(split)
    out_dir = config.CROPS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    grouped = df.groupby("image_name")

    for image_name, group in tqdm(grouped, desc=f"Cropping {split} bounding boxes"):
        img_path = config.RAW_DATA_DIR / config.SKU110K_IMAGE_SUBDIR / image_name
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        for i, (_, row) in enumerate(group.iterrows()):
            x1 = max(int(row["x1"]) - padding, 0)
            y1 = max(int(row["y1"]) - padding, 0)
            x2 = min(int(row["x2"]) + padding, w)
            y2 = min(int(row["y2"]) + padding, h)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{Path(image_name).stem}_{i:04d}.jpg"
            crop_path = out_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            manifest.append({
                "crop_file": crop_name,
                "source_image": image_name,
                "bbox": [x1, y1, x2, y2],
                "label": None,   # to be filled in via a labeled classification dataset
            })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] {len(manifest)} crops written to {out_dir}")
    print(f"[OK] Manifest written to {manifest_path}")
    print("[NOTE] 'label' fields are null - merge with a labeled product-category "
          "dataset (e.g. Freiburg Groceries / RPC) or manual annotation before "
          "training the classifier.")


def compute_dataset_stats(split: str) -> dict:
    """Quick sanity-check statistics for a split (box count, boxes/image, sizes)."""
    df = load_annotations(split)
    stats = {
        "num_images": df["image_name"].nunique(),
        "num_boxes": len(df),
        "avg_boxes_per_image": round(len(df) / df["image_name"].nunique(), 2),
        "avg_box_width": round((df["x2"] - df["x1"]).mean(), 2),
        "avg_box_height": round((df["y2"] - df["y1"]).mean(), 2),
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKU110K preprocessing for shelf detection + classification")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--crop", action="store_true", help="Also crop bounding boxes for the classifier")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics and exit")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for s in splits:
        if args.stats:
            print(s, compute_dataset_stats(s))
            continue

        convert_to_yolo_format(s)
        if args.crop:
            crop_bounding_boxes(s)
