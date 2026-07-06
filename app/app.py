"""
app.py
=======
Streamlit deployment for the Smart Retail Shelf Assistant.

Flow:
    1. User uploads a shelf photo.
    2. CNN pipeline (detection + classification) analyzes it.
    3. Results (annotated image + category counts + stock gaps) are shown.
    4. A chatbot interface lets the user ask follow-up questions
       ("Do you have any gluten-free snacks?", "What should I restock?",
       "Suggest an alternative to X") answered by the LLM, grounded in the
       CNN output via prompt engineering.

Run with:
    streamlit run app/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.pipeline import ShelfAssistantPipeline


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")


@st.cache_resource(show_spinner="Loading CNN + LLM pipeline ...")
def load_pipeline():
    try:
        return ShelfAssistantPipeline()
    except Exception as exc:
        st.error(f"Failed to initialize the pipeline: {exc}")
        raise


def draw_annotations(image: Image.Image, detections: list) -> Image.Image:
    """Overlay bounding boxes + predicted category labels on the shelf image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="#00C853", width=2)
        draw.text((x1, max(y1 - 12, 0)), det["label"], fill="#00C853")
    return annotated


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown(
        "Upload a retail shelf photo. A CNN detects and classifies every "
        "product, and an LLM-powered chatbot answers questions, generates "
        "restocking recommendations, and suggests alternatives."
    )
    st.divider()
    st.caption("Architecture")
    st.code(
        "Image -> Preprocessing -> CNN Detection\n"
        "      -> CNN Classification -> Prompt Engineering\n"
        "      -> LLM -> Explanation / Recommendation / Q&A",
        language="text",
    )
    st.divider()
    st.caption(f"CNN backbone: {config.CNN_BACKBONE}")
    st.caption(f"LLM provider: {config.LLM_PROVIDER} ({config.LLM_MODEL_NAME})")
    if st.button("🗑️ Reset conversation"):
        st.session_state.chat_history = []
        st.rerun()


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
col_image, col_chat = st.columns([1, 1])

with col_image:
    st.subheader("1. Upload a shelf image")
    uploaded_file = st.file_uploader("Shelf photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image

        if st.button("🔍 Analyze shelf", type="primary"):
            with st.spinner("Running CNN detection + classification ..."):
                pipeline = load_pipeline()
                temp_dir = Path(tempfile.gettempdir())
                temp_path = temp_dir / uploaded_file.name
                image.save(temp_path)
                st.session_state.analysis = pipeline.analyze_image(str(temp_path))
                st.session_state.chat_history = []  # reset chat for new image

    if st.session_state.uploaded_image is not None:
        if st.session_state.analysis is not None:
            annotated = draw_annotations(
                st.session_state.uploaded_image, st.session_state.analysis["detections"]
            )
            st.image(annotated, caption="Detected products", use_container_width=True)

            detections = st.session_state.analysis["detections"]
            gaps = st.session_state.analysis["gaps"]

            st.subheader("2. Shelf summary")
            counts = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1

            metric_cols = st.columns(3)
            metric_cols[0].metric("Products detected", len(detections))
            metric_cols[1].metric("Categories", len(counts))
            metric_cols[2].metric("Potential gaps", len(gaps))

            st.bar_chart(counts)

            if st.button("📝 Generate shelf report"):
                pipeline = load_pipeline()
                with st.spinner("Generating report ..."):
                    report = pipeline.generate_shelf_report(st.session_state.analysis)
                st.info(report)
        else:
            st.image(st.session_state.uploaded_image, caption="Uploaded image", use_container_width=True)
            st.caption("Click 'Analyze shelf' to run the CNN pipeline.")

with col_chat:
    st.subheader("3. Ask ShelfAI 🤖")
    st.caption(
        "Ask about stock levels, product categories, or request alternatives "
        "for items that look out of stock."
    )

    chat_container = st.container(height=420)
    with chat_container:
        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

    user_question = st.chat_input("e.g. 'Do you have any snacks left?' or 'Suggest an alternative to coffee'")

    if user_question:
        if st.session_state.analysis is None:
            st.warning("Please upload and analyze a shelf image first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            pipeline = load_pipeline()
            with st.spinner("Thinking ..."):
                # Trim history for the LLM context window
                history_for_llm = st.session_state.chat_history[-config.MAX_CHAT_HISTORY:]
                answer = pipeline.chat(
                    st.session_state.analysis, user_question, chat_history=history_for_llm
                )

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
