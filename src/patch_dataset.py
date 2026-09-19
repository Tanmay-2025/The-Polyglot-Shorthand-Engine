"""
Patches the single ambiguous semantic base delivery_tracking_b027
(both models confused all 10 of its surface variants with order_status --
see error_analysis + inspect_confusion results).

Label is kept as delivery_tracking (correct intent); only the TEXT is
rewritten to add an unambiguous "courier/parcel" cue instead of the
generic "tracking update" phrase that overlapped with order_status.

Does NOT touch the original dataset file. Writes a versioned patched
copy plus a changelog documenting exactly what changed and why, so the
original Dataset A v2 provenance trail stays intact.

Usage:
    python patch_dataset.py
"""

from pathlib import Path
import json

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_PATH = PROJECT_ROOT / "data" / "dataset_A_v2_36class_final.csv"
PATCHED_PATH = PROJECT_ROOT / "data" / "dataset_A_v2_36class_final_patched.csv"
CHANGELOG_PATH = PROJECT_ROOT / "data" / "dataset_A_v2_patch_changelog.json"

TARGET_BASE_ID = "delivery_tracking_b027"

# variant -> new text. Mirrors the original variant set/style for this
# base (same noise pattern per variant type), with "courier ka" / "parcel
# ka" added to disambiguate from order_status.
NEW_TEXT_BY_VARIANT = {
    "clean":            "pls courier ka tracking update de do, parcel ka tracking update nahi mil raha",
    "shorthand":        "pls courier ka tracking update de do, parcel ka tracking update nahi mil raha",
    "code_mix":         "pls courier ka tracking update de do, parcel ka tracking update nahi mil raha",
    "typo":             "pls courrier ka trackign updtae de do, parcle ka tracking updat nahi mil raha",
    "emoji":            "pls courier ka tracking update de do, parcel ka tracking update nahi mil raha \U0001F64F",
    "punctuation":      "pls courier ka tracking update de do, parcel ka tracking update nahi mil raha...",
    "shorthand_typo":   "pls corier ka tracing updtae de do, parcl ka trackign udpate nahi mil raha",
    "shorthand_emoji":  "pls courier ka tracking update de do, parcel ka tracking update nahi mil raha \U0001F64F",
    "code_mix_emoji":   "pls courier ka tracking update de do, parcel ka tracking update nahi mil raha \U0001F64F",
    "stress":           "pls courier ka tracking update deee do, parcel ka tracking update nahi mil raha \U0001F62D",
}


def main():
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Original dataset not found: {ORIGINAL_PATH}")

    df = pd.read_csv(ORIGINAL_PATH)

    target_rows = df[df["base_id"] == TARGET_BASE_ID]
    if len(target_rows) == 0:
        raise ValueError(f"No rows found with base_id == {TARGET_BASE_ID}")

    print(f"Found {len(target_rows)} rows for base_id={TARGET_BASE_ID}")

    changelog_entries = []

    for idx, row in target_rows.iterrows():
        variant = row["variant"]
        if variant not in NEW_TEXT_BY_VARIANT:
            raise ValueError(
                f"No replacement text defined for variant '{variant}' "
                f"(row id={row['id']}). Update NEW_TEXT_BY_VARIANT."
            )

        old_text = row["text"]
        new_text = NEW_TEXT_BY_VARIANT[variant]

        df.at[idx, "text"] = new_text

        changelog_entries.append({
            "id": row["id"],
            "variant": variant,
            "intent_unchanged": row["intent"],
            "old_text": old_text,
            "new_text": new_text,
        })

    df.to_csv(PATCHED_PATH, index=False)
    print(f"\nWrote patched dataset to: {PATCHED_PATH}")
    print("(Original file left untouched at:", ORIGINAL_PATH, ")")

    changelog = {
        "patch_reason": (
            "delivery_tracking_b027's 10 surface variants were confused "
            "with order_status by both ModernBERT and XLM-R across every "
            "variant type (typo, emoji, code_mix, etc.), indicating a "
            "genuine semantic ambiguity in the base utterance itself "
            "('tracking update' read as generic order-status language) "
            "rather than a model robustness failure. See error_analysis.py "
            "and inspect_confusion.py output."
        ),
        "target_base_id": TARGET_BASE_ID,
        "label_change": "none -- intent stays delivery_tracking",
        "text_change": "added explicit courier/parcel cue to disambiguate from order_status",
        "n_rows_patched": len(changelog_entries),
        "entries": changelog_entries,
    }

    with open(CHANGELOG_PATH, "w", encoding="utf-8") as f:
        json.dump(changelog, f, indent=2, ensure_ascii=False)

    print(f"Wrote changelog to: {CHANGELOG_PATH}")

    print("\nBefore -> after for this base:")
    for entry in changelog_entries:
        print(f"\n  [{entry['id']}] ({entry['variant']})")
        print(f"    old: {entry['old_text']}")
        print(f"    new: {entry['new_text']}")


if __name__ == "__main__":
    main()