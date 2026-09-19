from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

DATA_PATH = Path("data/dataset_A_v2_36class_final.csv")

MODELS = {
    "ModernBERT-base": "answerdotai/ModernBERT-base",
    "XLM-R-base": "FacebookAI/xlm-roberta-base",
}

THRESHOLDS = [64, 96, 128, 160, 256]


def percentile(values, p):
    return np.percentile(values, p)


def audit_model(name, model_id, texts):
    print("\n" + "=" * 70)
    print(name)
    print(model_id)
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    lengths = []

    for text in texts:
        tokens = tokenizer(
            str(text),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        lengths.append(len(tokens))

    lengths = np.array(lengths)

    print(f"Examples audited: {len(lengths)}")
    print(f"Min:  {lengths.min()}")
    print(f"P50:  {percentile(lengths, 50):.1f}")
    print(f"P90:  {percentile(lengths, 90):.1f}")
    print(f"P95:  {percentile(lengths, 95):.1f}")
    print(f"P99:  {percentile(lengths, 99):.1f}")
    print(f"Max:  {lengths.max()}")

    print("\nTruncation rates:")
    for threshold in THRESHOLDS:
        truncated = (lengths > threshold).sum()
        rate = truncated / len(lengths) * 100

        print(
            f"  max_length={threshold:3d}: "
            f"{truncated:5d} examples "
            f"({rate:6.2f}%)"
        )

    return lengths


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    # Use train + validation to make the preprocessing decision.
    # Test remains untouched for final evaluation.
    selection_df = df[df["split"].isin(["train", "validation"])]

    print("Dataset:", DATA_PATH)
    print("Total examples:", len(df))
    print("Train + validation examples:", len(selection_df))

    texts = selection_df["text"].astype(str).tolist()

    results = {}

    for name, model_id in MODELS.items():
        results[name] = audit_model(name, model_id, texts)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()