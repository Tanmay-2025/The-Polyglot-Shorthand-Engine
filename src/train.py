from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


# ============================================================
# Frozen experiment configuration
# ============================================================

# Paths are anchored to the project root (the parent of this file's
# folder, e.g. src/train.py -> project root is src/..), so this script
# works whether you run it from the project root or from inside src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "dataset_A_v2_36class_final.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MAX_LENGTH = 64
NUM_LABELS = 36

SEED = 42
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
GRAD_ACCUMULATION = 1

EARLY_STOPPING_PATIENCE = 2


MODELS = {
    "modernbert": {
        "name": "ModernBERT-base",
        "model_id": "answerdotai/ModernBERT-base",
        "precision": "fp16",
        "adam_epsilon": 1e-8,
    },
    "xlmr": {
        "name": "XLM-R-base",
        "model_id": "FacebookAI/xlm-roberta-base",
        "precision": "fp16",
        "adam_epsilon": 1e-8,
    },
    "mdeberta": {
        "name": "mDeBERTa-v3-base",
        "model_id": "microsoft/mdeberta-v3-base",
        "precision": "fp32",
        # DeBERTa-v3's disentangled attention produces large embedding-
        # scale values that combine badly with AdamW's default
        # epsilon=1e-8 (near-zero denominator -> NaN loss/grad_norm from
        # step 1, independent of fp16/bf16/fp32). This is a documented
        # DeBERTa-specific fix, not something specific to this dataset.
        "adam_epsilon": 1e-6,
        # Diagnostic bump: loss was pinned at ln(36)=3.58 (uniform-
        # random baseline) even after the NaN fix, suggesting the
        # classifier head isn't receiving useful gradient signal at the
        # standard 2e-5 LR. If this doesn't move loss off ln(36) either,
        # the issue is likely a transformers-version incompatibility
        # with DebertaV2, not a hyperparameter problem.
        "learning_rate": 1e-4,
    },
    "muril": {
        "name": "MuRIL-base-cased",
        "model_id": "google/muril-base-cased",
        "precision": "fp16",
        "adam_epsilon": 1e-8,
    },
}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# Dataset preparation
# ============================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "text",
        "intent",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    # Freeze label mapping alphabetically.
    intents = sorted(df["intent"].unique())

    if len(intents) != NUM_LABELS:
        raise ValueError(
            f"Expected {NUM_LABELS} intents, found {len(intents)}"
        )

    label2id = {
        label: idx
        for idx, label in enumerate(intents)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    df["label"] = df["intent"].map(label2id)

    train_df = df[df["split"] == "train"][
        ["text", "label"]
    ].reset_index(drop=True)

    val_df = df[df["split"] == "validation"][
        ["text", "label"]
    ].reset_index(drop=True)

    test_df = df[df["split"] == "test"][
        ["text", "label"]
    ].reset_index(drop=True)

    print("\nDataset sizes:")
    print("  Train:", len(train_df))
    print("  Validation:", len(val_df))
    print("  Test:", len(test_df))

    print("\nLabel mapping:")
    for idx in range(NUM_LABELS):
        print(f"  {idx:2d} -> {id2label[idx]}")

    return (
        Dataset.from_pandas(train_df, preserve_index=False),
        Dataset.from_pandas(val_df, preserve_index=False),
        Dataset.from_pandas(test_df, preserve_index=False),
        label2id,
        id2label,
    )


# ============================================================
# Tokenization
# ============================================================

def tokenize_datasets(
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizer,
):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        desc="Tokenizing train",
    )

    val_dataset = val_dataset.map(
        tokenize,
        batched=True,
        desc="Tokenizing validation",
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        desc="Tokenizing test",
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# TrainingArguments compatibility
# ============================================================

def build_training_args(output_dir, num_training_steps, precision, adam_epsilon, learning_rate):
    # Our frozen configuration specifies 10% warmup.
    warmup_steps = int(
        num_training_steps * WARMUP_RATIO
    )

    print("\nTraining configuration:")
    print(f"  Epochs:              {NUM_EPOCHS}")
    print(f"  Learning rate:       {learning_rate}")
    print(f"  Weight decay:        {WEIGHT_DECAY}")
    print(f"  Warmup ratio:        {WARMUP_RATIO}")
    print(f"  Warmup steps:        {warmup_steps}")
    print(f"  Train batch size:    {TRAIN_BATCH_SIZE}")
    print(f"  Eval batch size:     {EVAL_BATCH_SIZE}")
    print(f"  Max length:          {MAX_LENGTH}")
    print(f"  Precision:           {precision}")
    print(f"  Seed:                {SEED}")

    return TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=NUM_EPOCHS,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=GRAD_ACCUMULATION,

        fp16=(precision == "fp16" and torch.cuda.is_available()),
        bf16=(precision == "bf16" and torch.cuda.is_available()),

        adam_epsilon=adam_epsilon,

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

# ============================================================
# Main training function
# ============================================================

def train(model_key):
    config = MODELS[model_key]

    model_name = config["name"]
    model_id = config["model_id"]
    learning_rate = config.get("learning_rate", LEARNING_RATE)

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print(f"MODEL:    {model_id}")
    print("=" * 70)

    print("\nDevice:")
    print("  CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "  GPU:",
            torch.cuda.get_device_name(0)
        )

    seed_everything(SEED)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        label2id,
        id2label,
    ) = load_dataset()

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = tokenize_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    output_dir = OUTPUT_ROOT / model_key

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps_per_epoch = (
    len(train_dataset)
    // (
        TRAIN_BATCH_SIZE
        * GRAD_ACCUMULATION
    )
)

    num_training_steps = (
        steps_per_epoch * NUM_EPOCHS
)

    precision = config.get("precision", "fp16")
    adam_epsilon = config.get("adam_epsilon", 1e-8)

    training_args = build_training_args(
        output_dir,
        num_training_steps,
        precision,
        adam_epsilon,
        learning_rate,
)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

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

    train_result = trainer.train()

    print("\nTraining complete.")

    print("\nBest checkpoint:")
    print(trainer.state.best_model_checkpoint)

    print("\nBest validation macro-F1:")
    print(trainer.state.best_metric)

    print("\nFinal validation evaluation:")
    val_metrics = trainer.evaluate(
        eval_dataset=val_dataset
    )

    print(val_metrics)

    # ------------------------------------------------------------
    # Detach EarlyStoppingCallback before the final TEST evaluation.
    #
    # EarlyStoppingCallback.on_evaluate() runs on every trainer.evaluate()
    # call (validation OR test) and looks up args.metric_for_best_model
    # ("eval_macro_f1") in the returned metrics dict. The test pass is
    # run with metric_key_prefix="test", so the metrics dict only has
    # "test_macro_f1" -- the key the callback expects is never there,
    # which is exactly what produced the
    #   "early stopping required metric_for_best_model ... but did not
    #    find eval_macro_f1"
    # warning. This has no effect on the reported metrics (the callback
    # doesn't alter predictions/metrics, only decides whether to stop
    # training), but leaving it attached during a non-validation eval
    # is misleading, so we remove it before the test pass.
    # ------------------------------------------------------------
    if early_stopping_callback in trainer.callback_handler.callbacks:
        trainer.callback_handler.callbacks.remove(early_stopping_callback)

    print("\nFinal TEST evaluation:")
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )

    print(test_metrics)

    # Save model/tokenizer.
    final_dir = output_dir / "best_model"

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save experiment metadata.
    metadata = {
        "model_name": model_name,
        "model_id": model_id,
        "dataset": str(DATA_PATH),
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "precision": precision,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(
        output_dir / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nSaved to:")
    print(final_dir)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["modernbert", "xlmr", "mdeberta", "muril"],
        required=True,
    )

    args = parser.parse_args()

    train(args.model)


if __name__ == "__main__":
    main()