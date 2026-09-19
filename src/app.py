"""
Demo app for the intent classifier (and secondary sentiment classifier).
Loads the trained XLM-R models and lets you type Hinglish/code-mixed text
and see the prediction, confidence, and measured inference latency live.

Usage:
    pip install gradio
    python src/app.py

Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

from pathlib import Path
import time

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTENT_MODEL_DIR = PROJECT_ROOT / "outputs" / "xlmr" / "best_model"
SENTIMENT_MODEL_DIR = PROJECT_ROOT / "outputs" / "sentiment_xlmr" / "best_model"

MAX_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir):
    if not model_dir.exists():
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


print(f"Device: {DEVICE}")
print("Loading intent classifier (XLM-R)...")
intent_tokenizer, intent_model = load_model(INTENT_MODEL_DIR)

print("Loading sentiment classifier (XLM-R)...")
sentiment_tokenizer, sentiment_model = load_model(SENTIMENT_MODEL_DIR)


@torch.no_grad()
def classify(text, tokenizer, model):
    if not text or not text.strip():
        return None, None, None

    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    logits = model(**enc).logits

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id].item())

    # Top 3 for context
    top3_ids = torch.topk(probs, k=min(3, probs.shape[0])).indices.tolist()
    top3 = [(model.config.id2label[i], float(probs[i].item())) for i in top3_ids]

    return label, confidence, latency_ms, top3


def predict_intent(text):
    if intent_model is None:
        return "Model not found — train it first (see README).", "", ""
    label, confidence, latency_ms, top3 = classify(text, intent_tokenizer, intent_model)
    if label is None:
        return "", "", ""

    top3_str = "\n".join(f"  {i+1}. {lbl} ({p:.1%})" for i, (lbl, p) in enumerate(top3))
    result = f"**Predicted intent:** `{label}`  ({confidence:.1%} confidence)"
    latency_str = f"**Inference latency:** {latency_ms:.2f} ms  (device: {DEVICE})"
    return result, top3_str, latency_str


def predict_sentiment(text):
    if sentiment_model is None:
        return "Model not found — train it first (see README).", "", ""
    label, confidence, latency_ms, top3 = classify(text, sentiment_tokenizer, sentiment_model)
    if label is None:
        return "", "", ""

    top3_str = "\n".join(f"  {i+1}. {lbl} ({p:.1%})" for i, (lbl, p) in enumerate(top3))
    result = f"**Predicted sentiment:** `{label}`  ({confidence:.1%} confidence)"
    latency_str = f"**Inference latency:** {latency_ms:.2f} ms  (device: {DEVICE})"
    return result, top3_str, latency_str


INTENT_EXAMPLES = [
    "bhai mera order cancel krdo please",
    "delivery delayed h aur koi update nahi mila",
    "mera refund abhi tk nahi aaya",
    "app baar baar crash ho jata h 😤",
    "kisi insaan se baat krni h",
]

SENTIMENT_EXAMPLES = [
    "bohot badhiya service h aapki",
    "bohot badhiya service h aapki 😒",
    "order kal deliver hoga",
    "service bohot kharab h bilkul bhi acha nahi",
    "issue turant solve ho gya bohot badhiya 🙃",
]

with gr.Blocks(title="Polyglot's Shorthand — Demo") as demo:
    gr.Markdown("# The Polyglot's Shorthand — Live Demo")
    gr.Markdown(
        f"Intent classifier: **XLM-R-base**, trained on Dataset A v2 (36 intents). "
        f"Running on **{DEVICE}**. See `whitepaper.md` for full methodology and results."
    )

    with gr.Tab("Intent Classification (Primary)"):
        gr.Markdown("Type a Hinglish/code-mixed customer-support message.")
        intent_input = gr.Textbox(label="Input text", placeholder="e.g. bhai order cancel krdo please")
        intent_btn = gr.Button("Classify", variant="primary")
        intent_result = gr.Markdown()
        intent_latency = gr.Markdown()
        intent_top3 = gr.Textbox(label="Top 3 predictions", interactive=False)
        gr.Examples(examples=INTENT_EXAMPLES, inputs=intent_input)

        intent_btn.click(
            predict_intent,
            inputs=intent_input,
            outputs=[intent_result, intent_top3, intent_latency],
        )

    with gr.Tab("Sentiment Analysis (Secondary)"):
        gr.Markdown(
            "Tests whether emoji is read as semantic signal — try the same sentence "
            "with and without a sarcastic emoji (e.g. 😒 🙄 🙃)."
        )
        sentiment_input = gr.Textbox(label="Input text", placeholder="e.g. bohot badhiya service h aapki")
        sentiment_btn = gr.Button("Classify", variant="primary")
        sentiment_result = gr.Markdown()
        sentiment_latency = gr.Markdown()
        sentiment_top3 = gr.Textbox(label="Top 3 predictions", interactive=False)
        gr.Examples(examples=SENTIMENT_EXAMPLES, inputs=sentiment_input)

        sentiment_btn.click(
            predict_sentiment,
            inputs=sentiment_input,
            outputs=[sentiment_result, sentiment_top3, sentiment_latency],
        )

if __name__ == "__main__":
    demo.launch()