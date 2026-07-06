"""
prompt_engineering.py
======================
Prompt templates that turn structured CNN output (detected products, counts,
categories, shelf gaps) into well-formed prompts for the LLM.

Keeping prompts in one place makes them easy to version, A/B test, and reuse
across the Streamlit chatbot, the report-generation function, and any batch
/offline analytics job.
"""

from typing import List, Dict


SYSTEM_PROMPT = """\
You are "ShelfAI", a helpful and concise retail assistant embedded in a store's
Smart Shelf monitoring system. You are given structured data extracted by a
computer-vision pipeline (a CNN object detector + product classifier) that has
just analyzed a photo of a retail shelf. Your job is to:

1. Explain, in plain language, what is on the shelf.
2. Flag any stock issues (empty gaps, low counts of a category).
3. Answer customer or store-manager questions about the products shown.
4. Recommend alternative or complementary products when asked, or when a
   product appears to be out of stock.
5. Keep answers concise, friendly, and grounded ONLY in the shelf data you
   were given - never invent products that are not in the data.

If asked something the shelf data cannot answer, say so plainly instead of
guessing.
"""


def format_shelf_context(detections: List[Dict]) -> str:
    """
    Turn a list of {"label": str, "confidence": float, "bbox": [...]}
    dicts (from ProductClassifier + ShelfProductDetector) into a compact,
    LLM-friendly summary block.
    """
    if not detections:
        return "No products were detected on this shelf image."

    counts: Dict[str, int] = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    lines = [f"Total products detected: {len(detections)}", "Category breakdown:"]
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"  - {label}: {count} unit(s)")

    return "\n".join(lines)


def format_gap_context(gaps: List[Dict]) -> str:
    if not gaps:
        return "No significant shelf gaps detected."
    lines = [f"{len(gaps)} potential low-stock / empty gap(s) detected on the shelf:"]
    for g in gaps:
        lines.append(f"  - gap of {g['gap_width_px']}px between x={g['gap_start_x']} and x={g['gap_end_x']}")
    return "\n".join(lines)


def build_shelf_report_prompt(detections: List[Dict], gaps: List[Dict]) -> str:
    """Prompt used to auto-generate a natural-language shelf status report."""
    return f"""\
{format_shelf_context(detections)}

{format_gap_context(gaps)}

Task: Write a short shelf status report (4-6 sentences) for a store manager.
Mention what is well-stocked, what looks low/empty, and one concrete
restocking recommendation if a gap was detected.
"""


def build_recommendation_prompt(detections: List[Dict], requested_product: str) -> str:
    """Prompt used when a customer asks for alternatives to an item."""
    return f"""\
{format_shelf_context(detections)}

Customer request: The customer is looking for "{requested_product}" and it may
not be visible in the shelf data above.

Task: Based only on the categories actually present on this shelf, suggest up
to 3 reasonable alternative or complementary products the customer could
consider instead. If nothing relevant is present, say so honestly.
"""


def build_chat_prompt(detections: List[Dict], gaps: List[Dict], user_question: str,
                       chat_history: List[Dict] = None) -> List[Dict]:
    """
    Builds the full OpenAI-style `messages` list for a single chatbot turn,
    combining the system prompt, shelf context, prior turns, and the new
    user question. This is the function called by app.py on every message.
    """
    context_block = f"""\
[SHELF CONTEXT - do not repeat this block verbatim to the user]
{format_shelf_context(detections)}

{format_gap_context(gaps)}
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context_block}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_question})
    return messages
