"""
Trains XLM-R-base on the 3-class sentiment dataset (dataset_sentiment_v1.csv).

Reuses the winning model from the intent-classification comparison
(XLM-R) rather than re-running a model comparison for this task. Dataset
is small (150 hand-authored rows) so this does its own stratified
train/val/test split at run-time rather than relying on a pre-split file.

Usage:
    python train_sentiment.py
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_sentiment_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sentiment_xlmr"

MODEL_ID = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 64
NUM_LABELS = 3

SEED = 42
# More epochs than the intent-classification run (5) since this dataset
# is much smaller (105 train rows vs 15,120) -- fewer gradient steps per
# epoch means more passes are needed to converge.
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRAD_ACCUMULATION = 1

EARLY_STOPPING_PATIENCE = 3

# Stratified split ratios.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def load_and_split():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    labels = sorted(df["sentiment"].unique())
    if len(labels) != NUM_LABELS:
        raise ValueError(f"Expected {NUM_LABELS} sentiment classes, found {len(labels)}: {labels}")

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["sentiment"].map(label2id)

    # Stratified 70/15/15 split.
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df["sentiment"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=temp_df["sentiment"],
        random_state=SEED,
    )

    for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"  {name}: {len(split_df)}  ({split_df['sentiment'].value_counts().to_dict()})")

    cols = ["text", "label"]
    return (
        Dataset.from_pandas(train_df[cols].reset_index(drop=True)),
        Dataset.from_pandas(val_df[cols].reset_index(drop=True)),
        Dataset.from_pandas(test_df[cols].reset_index(drop=True)),
        label2id,
        id2label,
        train_df, val_df, test_df,
    )


def main():
    print("=" * 70)
    print("TRAINING: XLM-R-base on 3-class sentiment")
    print("=" * 70)

    print("\nDevice:")
    print("  CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  GPU:", torch.cuda.get_device_name(0))

    seed_everything(SEED)

    print("\nLoading and splitting dataset (stratified 70/15/15)...")
    (
        train_dataset, val_dataset, test_dataset,
        label2id, id2label,
        train_df, val_df, test_df,
    ) = load_and_split()

    print("\nLabel mapping:")
    for idx in range(NUM_LABELS):
        print(f"  {idx} -> {id2label[idx]}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize, batched=True, desc="Tokenizing train")
    val_dataset = val_dataset.map(tokenize, batched=True, desc="Tokenizing validation")
    test_dataset = test_dataset.map(tokenize, batched=True, desc="Tokenizing test")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, len(train_dataset) // (TRAIN_BATCH_SIZE * GRAD_ACCUMULATION))
    num_training_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(num_training_steps * WARMUP_RATIO)

    print("\nTraining configuration:")
    print(f"  Epochs:           {NUM_EPOCHS}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"  Warmup steps:     {warmup_steps}")
    print(f"  FP16:             {torch.cuda.is_available()}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        fp16=torch.cuda.is_available(),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    print("\nStarting training...\n")
    trainer.train()

    print("\nBest checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation macro-F1:", trainer.state.best_metric)

    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("\nFinal validation evaluation:", val_metrics)

    # Detach early stopping before final test eval (same reasoning as
    # the main train.py pipeline).
    if early_stopping_callback in trainer.callback_handler.callbacks:
        trainer.callback_handler.callbacks.remove(early_stopping_callback)

    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print("\nFinal TEST evaluation:", test_metrics)

    final_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save test predictions for error inspection.
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    test_df = test_df.copy().reset_index(drop=True)
    test_df["predicted_sentiment"] = [id2label[p] for p in pred_labels]
    test_df["correct"] = test_df["sentiment"] == test_df["predicted_sentiment"]
    test_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    metadata = {
        "model_id": MODEL_ID,
        "dataset": str(DATA_PATH),
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to: {final_dir}")
    print(f"Test predictions: {OUTPUT_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()