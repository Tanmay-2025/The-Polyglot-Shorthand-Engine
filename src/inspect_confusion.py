"""
Pulls the delivery_tracking -> order_status misclassified rows from both
models' error_analysis_misclassified.csv and prints them side by side for
manual inspection.

Usage:
    python inspect_confusion.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_errors(model_key):
    path = PROJECT_ROOT / "outputs" / model_key / "error_analysis_misclassified.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run error_analysis.py for {model_key} first."
        )
    return pd.read_csv(path)


def main():
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)

    modernbert_errors = load_errors("modernbert")
    xlmr_errors = load_errors("xlmr")

    mb_confusion = modernbert_errors[
        (modernbert_errors["intent"] == "delivery_tracking")
        & (modernbert_errors["predicted_intent"] == "order_status")
    ]

    xlmr_confusion = xlmr_errors[
        (xlmr_errors["intent"] == "delivery_tracking")
        & (xlmr_errors["predicted_intent"] == "order_status")
    ]

    print(f"ModernBERT: {len(mb_confusion)} delivery_tracking -> order_status errors")
    print(f"XLM-R:      {len(xlmr_confusion)} delivery_tracking -> order_status errors\n")

    cols = ["id", "base_id", "text", "variant", "noise_level"]

    print("=" * 100)
    print("MODERNBERT: delivery_tracking -> order_status")
    print("=" * 100)
    for _, row in mb_confusion[cols].iterrows():
        print(f"\n[{row['id']}]  variant={row['variant']}  noise_level={row['noise_level']}")
        print(f"  text: {row['text']}")

    print("\n" + "=" * 100)
    print("XLM-R: delivery_tracking -> order_status")
    print("=" * 100)
    for _, row in xlmr_confusion[cols].iterrows():
        print(f"\n[{row['id']}]  variant={row['variant']}  noise_level={row['noise_level']}")
        print(f"  text: {row['text']}")

    # Check overlap: are both models failing on the exact same base utterances?
    mb_base_ids = set(mb_confusion["base_id"])
    xlmr_base_ids = set(xlmr_confusion["base_id"])
    shared_base_ids = mb_base_ids & xlmr_base_ids

    print("\n" + "=" * 100)
    print("OVERLAP CHECK")
    print("=" * 100)
    print(f"ModernBERT unique base_ids in this confusion: {len(mb_base_ids)}")
    print(f"XLM-R unique base_ids in this confusion:      {len(xlmr_base_ids)}")
    print(f"Shared base_ids (both models fail on):         {len(shared_base_ids)}")
    if shared_base_ids:
        print(f"  -> {sorted(shared_base_ids)}")


if __name__ == "__main__":
    main()
    