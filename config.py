"""
config.py
=========
Central configuration for the Smart Retail Shelf Assistant project.

All paths, hyperparameters, and API settings are defined here so that
every module (data pipeline, training scripts, inference pipeline, and
the Streamlit app) reads from a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "SKU110K"          # downloaded SKU110K dataset root
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CROPS_DIR = PROCESSED_DATA_DIR / "crops"              # cropped product images (for classifier)
YOLO_DATA_DIR = PROCESSED_DATA_DIR / "yolo_format"    # SKU110K converted to YOLO txt format

MODELS_DIR = ROOT_DIR / "models"
DETECTION_WEIGHTS = MODELS_DIR / "shelf_detector_yolov8.pt"
CLASSIFIER_WEIGHTS = MODELS_DIR / "product_classifier_resnet50.pt"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"

LOGS_DIR = ROOT_DIR / "logs"

# ------------------------------------------------------------------
# SKU110K dataset info
# https://github.com/eg4000/SKU110K_CVPR19
# ------------------------------------------------------------------
SKU110K_URL = "https://github.com/eg4000/SKU110K_CVPR19"
SKU110K_IMAGE_SUBDIR = "images"
SKU110K_ANNOTATION_TRAIN = "annotations_train.csv"
SKU110K_ANNOTATION_VAL = "annotations_val.csv"
SKU110K_ANNOTATION_TEST = "annotations_test.csv"
# SKU110K annotation columns (no header in the original CSV):
SKU110K_COLUMNS = [
    "image_name", "x1", "y1", "x2", "y2",
    "class", "image_width", "image_height",
]

# ------------------------------------------------------------------
# Detection model (Stage 1: locate every product on the shelf)
# ------------------------------------------------------------------
# Two interchangeable detection backends are supported:
#   "retinanet_pretrained" -> loads the ready-trained SKU110K RetinaNet
#                              checkpoint (models/iou_resnet50_csv_06.h5),
#                              no training needed. Requires TensorFlow.
#   "yolov8"               -> loads/YOLOv8 model. This is the default
#                              backend and works on Python 3.14 without
#                              TensorFlow.
DETECTION_BACKEND = "yolov8"

# --- RetinaNet (SKU110K official architecture: ResNet50 + FPN + IoU-Net) ---
# This is the architecture produced by the eg4000/SKU110K_CVPR19 training
# script (a fork of fizyr/keras-retinanet with an added IoU-prediction head).
# Checkpoint naming convention: "iou_resnet50_csv_<epoch>.h5"
RETINANET_WEIGHTS = MODELS_DIR / "iou_resnet50_csv_06.h5"
RETINANET_BACKBONE = "resnet50"
RETINANET_PYRAMID_LEVELS = [3, 4, 5, 6, 7]          # P3..P7
RETINANET_ANCHOR_SIZES = [32, 64, 128, 256, 512]     # one per pyramid level
RETINANET_ANCHOR_STRIDES = [8, 16, 32, 64, 128]      # one per pyramid level
RETINANET_ANCHOR_RATIOS = [0.5, 1.0, 2.0]
RETINANET_ANCHOR_SCALES = [2 ** 0.0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
RETINANET_MAX_IMAGE_SIDE = 1333
RETINANET_MIN_IMAGE_SIDE = 800
RETINANET_SCORE_THRESHOLD = 0.5      # classification score cutoff before NMS
RETINANET_NMS_IOU_THRESHOLD = 0.45
RETINANET_MAX_DETECTIONS = 500       # SKU110K shelves are dense; keep this high
RETINANET_USE_IOU_HEAD = True        # combine classification score with the
                                      # IoU-Net confidence head, score = cls * iou
                                      # (simplified stand-in for the paper's
                                      # full EM-merger; see detection_model.py)

# --- YOLOv8 (only used if DETECTION_BACKEND == "yolov8") ---
DETECTION_MODEL_ARCH = "yolov8n.pt"      # nano backbone; swap for yolov8s/m for higher accuracy
DETECTION_IMG_SIZE = 640
DETECTION_EPOCHS = 50
DETECTION_BATCH_SIZE = 16
DETECTION_CONF_THRESHOLD = 0.35
DETECTION_IOU_THRESHOLD = 0.45

# ------------------------------------------------------------------
# Classification model (Stage 2: classify each cropped product)
# ------------------------------------------------------------------
CNN_BACKBONE = "resnet50"          # one of: resnet50 | efficientnet_b0 | mobilenet_v3_large | vgg16
CNN_IMG_SIZE = 224
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 25
CNN_LEARNING_RATE = 1e-4
CNN_NUM_WORKERS = 4
FREEZE_BACKBONE_EPOCHS = 5          # epochs to train only the classifier head before fine-tuning

# Product category taxonomy used for the classifier head.
# SKU110K itself is single-class (only "object"), so this taxonomy is the
# label set for a paired product-category dataset (e.g. Freiburg Groceries,
# RPC Retail Product Checkout, or a store-specific labeled subset).
DEFAULT_CLASSES = [
    "beverages", "snacks", "dairy", "bakery", "cereal",
    "canned_goods", "condiments", "frozen_foods", "personal_care",
    "household_cleaning", "confectionery", "fresh_produce",
]

# ------------------------------------------------------------------
# LLM / GenAI configuration
# ------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # openai | gemini | mistral | llama_local
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 512

# ------------------------------------------------------------------
# Vector DB (optional RAG over product catalog / planogram / FAQ)
# ------------------------------------------------------------------
VECTOR_DB_DIR = ROOT_DIR / "data" / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------------
APP_TITLE = "Smart Retail Shelf Assistant"
APP_ICON = "🛒"
MAX_CHAT_HISTORY = 20

for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CROPS_DIR, YOLO_DATA_DIR,
             MODELS_DIR, LOGS_DIR, VECTOR_DB_DIR]:
    os.makedirs(_dir, exist_ok=True)
