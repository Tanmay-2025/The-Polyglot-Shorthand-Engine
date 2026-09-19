"""
Final evaluation against Dataset B -- the held-out robustness benchmark.

Dataset B was NOT used for any tuning, model selection, or hyperparameter
decisions. It is hand-authored independently of Dataset A's
base-utterance + variant-transform generation process, so it is a
genuinely out-of-distribution check rather than a re-test of the same
generative patterns Dataset A already covers.

Usage:
    python eval_dataset_B.py --model_dir outputs/modernbert/best_model --device cuda
    python eval_dataset_B.py --model_dir outputs/xlmr/best_model --device cuda
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_B_PATH = PROJECT_ROOT / "data" / "dataset_B_robustness.csv"
MAX_LENGTH = 64
PREDICT_BATCH_SIZE = 64


@torch.no_grad()
def predict_all(model, tokenizer, texts, device, id2label):
    model.eval()
    preds = []

    for i in range(0, len(texts), PREDICT_BATCH_SIZE):
        batch = texts[i : i + PREDICT_BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
        preds.extend(batch_preds.tolist())

    return [id2label[p] for p in preds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")

    if not DATASET_B_PATH.exists():
        raise FileNotFoundError(f"Dataset B not found: {DATASET_B_PATH}")

    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    id2label = model.config.id2label
    id2label = {int(k): v for k, v in id2label.items()}

    df = pd.read_csv(DATASET_B_PATH)
    print(f"\nDataset B: {len(df)} examples, {df['intent'].nunique()} intents")

    texts = df["text"].astype(str).tolist()
    true_labels = df["intent"].tolist()

    print("\nRunning predictions...")
    pred_labels = predict_all(model, tokenizer, texts, device, id2label)

    df = df.copy()
    df["predicted_intent"] = pred_labels
    df["correct"] = df["intent"] == df["predicted_intent"]

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print(f"\nDataset B accuracy:    {accuracy:.4%}")
    print(f"Dataset B macro-F1:    {macro_f1:.4%}")
    print(f"Dataset B weighted-F1: {weighted_f1:.4%}")

    errors_df = df[~df["correct"]]
    print(f"\nErrors: {len(errors_df)} / {len(df)}")

    if len(errors_df) > 0:
        print("\nMisclassified examples:")
        for _, row in errors_df.iterrows():
            print(f"  [{row['id']}] true={row['intent']:30s} pred={row['predicted_intent']:30s}  text: {row['text']}")

    # Error rate by domain, for comparison against the Dataset A test-set
    # domain breakdown in error_analysis.py.
    domain_counts = df.groupby("domain").size()
    domain_error_counts = errors_df.groupby("domain").size() if len(errors_df) else pd.Series(dtype=int)

    domain_breakdown = []
    for domain in sorted(domain_counts.index):
        total = int(domain_counts[domain])
        errs = int(domain_error_counts.get(domain, 0))
        domain_breakdown.append({
            "domain": domain,
            "total": total,
            "errors": errs,
            "error_rate": errs / total if total else 0.0,
        })

    domain_breakdown.sort(key=lambda x: -x["error_rate"])

    print("\nError rate by domain:")
    for row in domain_breakdown:
        print(f"  {row['domain']:25s}  {row['errors']} / {row['total']}  = {row['error_rate']:.2%}")

    result = {
        "model_dir": str(model_dir),
        "dataset": "dataset_B_robustness (held-out, hand-authored, not used for tuning)",
        "n_examples": len(df),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_errors": int(len(errors_df)),
        "domain_breakdown": domain_breakdown,
        "misclassified": errors_df[["id", "text", "intent", "predicted_intent", "domain"]].to_dict(orient="records"),
    }

    out_path = model_dir.parent / "dataset_B_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()