"""
pipeline.py
============
End-to-end orchestration:

    Image Input
        -> Image Preprocessing
        -> CNN Detection (locate every product + shelf gaps)
        -> CNN Classification (category per detected product)
        -> Prompt Engineering
        -> LLM
        -> Generated explanation / recommendation / chat answer

This is the module imported by the Streamlit app (app/app.py). It is the
"glue" layer described in the assignment's architecture diagram.
"""

from pathlib import Path
from typing import List, Dict

from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.detection_model import build_shelf_detector
from src.classification_model import ProductClassifier
from src.llm_integration import LLMClient
from src import prompt_engineering as pe


class ShelfAssistantPipeline:
    """
    High-level facade used by the app. Loads the detector, classifier, and
    LLM client once, then exposes simple methods for a single image + chat turn.

    The detector backend is selected by config.DETECTION_BACKEND:
        "retinanet_pretrained" (default) - uses the ready-trained SKU110K
            RetinaNet checkpoint (models/iou_resnet50_csv_06.h5), no
            training needed.
        "yolov8" - trains/loads a YOLOv8 model instead (see train_detection.py).
    """

    def __init__(self):
        self.detector = build_shelf_detector()
        self.classifier = ProductClassifier()
        self.llm = LLMClient()

    # ------------------------------------------------------------------
    # Stage 1 + 2: CNN detection + classification
    # ------------------------------------------------------------------
    def analyze_image(self, image_path: str) -> Dict:
        """
        Runs detection -> per-box classification -> gap analysis on a shelf image.

        Returns:
            {
              "detections": [{"bbox":[...], "confidence":.., "label":.., "class_confidence":..}, ...],
              "gaps": [{"gap_start_x":.., "gap_end_x":.., "gap_width_px":..}, ...],
              "image_size": (width, height),
            }
        """
        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size

        raw_detections = self.detector.detect(image_path)

        enriched_detections = []
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = pil_image.crop((x1, y1, x2, y2))
            if crop.width == 0 or crop.height == 0:
                continue

            cls_result = self.classifier.predict(crop)
            enriched_detections.append({
                "bbox": det["bbox"],
                "detection_confidence": det["confidence"],
                "label": cls_result["label"],
                "class_confidence": cls_result["confidence"],
                "top3": cls_result["top3"],
            })

        gaps = self.detector.estimate_shelf_gaps(image_path, width, height)

        return {
            "detections": enriched_detections,
            "gaps": gaps,
            "image_size": (width, height),
        }

    # ------------------------------------------------------------------
    # Stage 3 + 4: Prompt engineering + LLM
    # ------------------------------------------------------------------
    def generate_shelf_report(self, analysis: Dict) -> str:
        """Stage 4 output: 'Description' - a natural-language shelf status report."""
        prompt = pe.build_shelf_report_prompt(analysis["detections"], analysis["gaps"])
        messages = [
            {"role": "system", "content": pe.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.llm.chat(messages)

    def recommend_alternatives(self, analysis: Dict, requested_product: str) -> str:
        """Stage 4 output: 'Recommendations' - alternatives for an out-of-stock/absent item."""
        prompt = pe.build_recommendation_prompt(analysis["detections"], requested_product)
        messages = [
            {"role": "system", "content": pe.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.llm.chat(messages)

    def chat(self, analysis: Dict, user_question: str, chat_history: List[Dict] = None) -> str:
        """Stage 4 output: 'Q&A' - free-form chatbot turn grounded in the shelf analysis."""
        messages = pe.build_chat_prompt(
            analysis["detections"], analysis["gaps"], user_question, chat_history
        )
        return self.llm.chat(messages)


if __name__ == "__main__":
    pipeline = ShelfAssistantPipeline()
    result = pipeline.analyze_image("data/raw/SKU110K/images/sample_shelf.jpg")
    print(f"Detected {len(result['detections'])} products, {len(result['gaps'])} gaps")
    print(pipeline.generate_shelf_report(result))
