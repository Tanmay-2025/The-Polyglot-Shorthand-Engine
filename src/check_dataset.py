from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/dataset_A_v2_36class_final.csv")

EXPECTED_ROWS = 21_600
EXPECTED_INTENTS = 36
EXPECTED_PER_INTENT = 600

EXPECTED_SPLITS = {
    "train": 15_120,
    "validation": 3_240,
    "test": 3_240,
}

REQUIRED_COLUMNS = {
    "id",
    "text",
    "intent",
    "variant",
    "source",
    "split",
}


def normalize(text):
    return " ".join(str(text).lower().split())


def main():
    print(f"Loading: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("\n=== BASIC SHAPE ===")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    assert len(df) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    )

    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    print("\n=== INTENTS ===")
    intents = sorted(df["intent"].unique())
    print("Number of intents:", len(intents))

    assert len(intents) == EXPECTED_INTENTS

    counts = df["intent"].value_counts()
    print("Min examples/intent:", counts.min())
    print("Max examples/intent:", counts.max())

    assert (counts == EXPECTED_PER_INTENT).all()

    print("\n=== SPLITS ===")
    split_counts = df["split"].value_counts().to_dict()
    print(split_counts)

    assert split_counts == EXPECTED_SPLITS

    print("\n=== IDS ===")
    assert df["id"].is_unique, "Duplicate IDs found"
    print("All IDs unique: YES")

    print("\n=== EMPTY TEXT ===")
    empty = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    print("Empty texts:", int(empty.sum()))

    assert not empty.any()

    print("\n=== SOURCE ===")
    print(df["source"].value_counts().to_dict())

    print("\n=== VARIANTS ===")
    print(df["variant"].value_counts().sort_index())

    print("\n=== NORMALIZED TEXT OVERLAP ===")

    df["_norm"] = df["text"].map(normalize)

    train_texts = set(df.loc[df["split"] == "train", "_norm"])
    val_texts = set(df.loc[df["split"] == "validation", "_norm"])
    test_texts = set(df.loc[df["split"] == "test", "_norm"])

    train_val = train_texts & val_texts
    train_test = train_texts & test_texts
    val_test = val_texts & test_texts

    print("Train ∩ Validation:", len(train_val))
    print("Train ∩ Test:", len(train_test))
    print("Validation ∩ Test:", len(val_test))

    assert len(train_val) == 0
    assert len(train_test) == 0
    assert len(val_test) == 0

    print("\n=== RESULT ===")
    print("DATASET CHECK PASSED")


if __name__ == "__main__":
    main()