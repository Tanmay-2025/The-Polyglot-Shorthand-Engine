"""
Error analysis for the trained intent classifiers.

Goes beyond aggregate test macro-F1 to answer:
  - which specific intent pairs get confused (confusion matrix -> top
    off-diagonal pairs)
  - whether errors cluster in particular surface-variant types
    (typo, code_mix, emoji, shorthand, stress, ...) -- this maps
    directly to the problem statement's robustness requirement
  - whether errors cluster in particular domains
  - the actual misclassified examples, for manual inspection

Usage:
    python error_analysis.py --model_dir outputs/modernbert/best_model --device cuda
    python error_analysis.py --model_dir outputs/xlmr/best_model --device cuda
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "dataset_A_v2_36class_final.csv"
MAX_LENGTH = 64
PREDICT_BATCH_SIZE = 64


def load_test_df(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(test_df) == 0:
        raise ValueError("No rows with split == 'test' found.")

    return test_df


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
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to a dataset CSV, e.g. the patched file. "
             "Defaults to data/dataset_A_v2_36class_final.csv.",
    )
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")

    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    id2label = model.config.id2label
    # HF sometimes loads id2label keys as strings; normalize to int keys.
    id2label = {int(k): v for k, v in id2label.items()}

    print(f"\nLoading test split from: {data_path}")
    test_df = load_test_df(data_path)
    print(f"Test examples: {len(test_df)}")

    texts = test_df["text"].astype(str).tolist()
    true_labels = test_df["intent"].tolist()

    print("\nRunning predictions...")
    pred_labels = predict_all(model, tokenizer, texts, device, id2label)

    test_df = test_df.copy()
    test_df["predicted_intent"] = pred_labels
    test_df["correct"] = test_df["intent"] == test_df["predicted_intent"]

    overall_acc = test_df["correct"].mean()
    n_errors = (~test_df["correct"]).sum()
    print(f"\nOverall test accuracy: {overall_acc:.4%}  ({n_errors} errors out of {len(test_df)})")

    errors_df = test_df[~test_df["correct"]].copy()

    # ------------------------------------------------------------
    # 1. Confusion matrix -> top confused intent pairs
    # ------------------------------------------------------------
    labels_sorted = sorted(test_df["intent"].unique())
    cm = confusion_matrix(test_df["intent"], test_df["predicted_intent"], labels=labels_sorted)

    confused_pairs = []
    for i, true_label in enumerate(labels_sorted):
        for j, pred_label in enumerate(labels_sorted):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_intent": true_label,
                    "predicted_intent": pred_label,
                    "count": int(cm[i, j]),
                })

    confused_pairs.sort(key=lambda x: -x["count"])

    print("\nTop confused intent pairs (true -> predicted):")
    for pair in confused_pairs[:15]:
        print(f"  {pair['true_intent']:35s} -> {pair['predicted_intent']:35s}  ({pair['count']} cases)")

    # ------------------------------------------------------------
    # 2. Errors by surface-variant type (typo, emoji, code_mix, ...)
    # ------------------------------------------------------------
    variant_counts = test_df.groupby("variant").size()
    variant_error_counts = errors_df.groupby("variant").size() if len(errors_df) else pd.Series(dtype=int)

    variant_breakdown = []
    for variant in sorted(variant_counts.index):
        total = int(variant_counts[variant])
        errs = int(variant_error_counts.get(variant, 0))
        variant_breakdown.append({
            "variant": variant,
            "total": total,
            "errors": errs,
            "error_rate": errs / total if total else 0.0,
        })

    variant_breakdown.sort(key=lambda x: -x["error_rate"])

    print("\nError rate by surface-variant type (highest first):")
    for row in variant_breakdown:
        print(f"  {row['variant']:20s}  {row['errors']:3d} / {row['total']:4d}  "
              f"= {row['error_rate']:.4%}")

    # ------------------------------------------------------------
    # 3. Errors by domain
    # ------------------------------------------------------------
    domain_counts = test_df.groupby("domain").size()
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

    print("\nError rate by domain (highest first):")
    for row in domain_breakdown:
        print(f"  {row['domain']:25s}  {row['errors']:3d} / {row['total']:4d}  "
              f"= {row['error_rate']:.4%}")

    # ------------------------------------------------------------
    # 4. Errors by intent (which classes are weakest)
    # ------------------------------------------------------------
    intent_counts = test_df.groupby("intent").size()
    intent_error_counts = errors_df.groupby("intent").size() if len(errors_df) else pd.Series(dtype=int)

    intent_breakdown = []
    for intent in sorted(intent_counts.index):
        total = int(intent_counts[intent])
        errs = int(intent_error_counts.get(intent, 0))
        intent_breakdown.append({
            "intent": intent,
            "total": total,
            "errors": errs,
            "error_rate": errs / total if total else 0.0,
        })

    intent_breakdown.sort(key=lambda x: -x["error_rate"])

    print("\nError rate by intent (highest first, top 15):")
    for row in intent_breakdown[:15]:
        print(f"  {row['intent']:30s}  {row['errors']:3d} / {row['total']:4d}  "
              f"= {row['error_rate']:.4%}")

    # ------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------
    out_dir = model_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model_dir": str(model_dir),
        "test_examples": len(test_df),
        "overall_accuracy": float(overall_acc),
        "n_errors": int(n_errors),
        "confused_pairs": confused_pairs,
        "variant_breakdown": variant_breakdown,
        "domain_breakdown": domain_breakdown,
        "intent_breakdown": intent_breakdown,
    }

    summary_path = out_dir / "error_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    errors_csv_path = out_dir / "error_analysis_misclassified.csv"
    errors_df[
        ["id", "base_id", "text", "domain", "intent", "predicted_intent", "variant", "noise_level", "emoji_role"]
    ].to_csv(errors_csv_path, index=False)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved misclassified examples to: {errors_csv_path}")


if __name__ == "__main__":
    main()