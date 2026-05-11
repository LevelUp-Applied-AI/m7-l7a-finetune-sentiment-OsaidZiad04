"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os

import dill
import numpy as np
import pandas as pd
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# Compatibility shim: datasets<=2.x expects an older dill/pickle signature,
# which breaks under Python 3.14 when building Dataset fingerprints.
try:
    from datasets.utils._dill import Pickler as _DatasetsPickler

    def _compat_batch_setitems(self, items, obj=None):
        if getattr(self, "_legacy_no_dict_keys_sorting", False):
            try:
                return super(_DatasetsPickler, self)._batch_setitems(items, obj)
            except TypeError:
                return super(_DatasetsPickler, self)._batch_setitems(items)
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        
        try:
            return dill.Pickler._batch_setitems(self, items, obj)
        except TypeError:
            return dill.Pickler._batch_setitems(self, items)

    _DatasetsPickler._batch_setitems = _compat_batch_setitems
except Exception:
    pass


# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    The CSV must have at least `text` and `label` columns. (The curated
    `data/app_reviews_train.csv` also includes `app`, `app_name`, and `rating`
    columns — these are useful for inspection but not required by the model.)

    Returns a `DatasetDict` with "train" and "test" keys.
    """
    df = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.train_test_split(test_size=test_size, seed=seed)


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    `tokenizer` is a loaded HuggingFace tokenizer (callable) — load it once
    in `main()` via `AutoTokenizer.from_pretrained(...)` and pass it in.
    Use truncation=True and max_length=max_length. Do not pad here — padding is
    applied dynamically by DataCollatorWithPadding at training time.

    Note: this signature differs from the drill (`tokenize_dataset(ds, name)`)
    by accepting the loaded tokenizer object so `main()` doesn't re-load it.
    """
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )
    args.eval_strategy = "epoch"
    args.save_strategy = "epoch"
    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids

    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer (trainer.model is the fine-tuned model). Pass
    id2label=ID2LABEL and label2id=LABEL2ID to the model so its config records
    the human-readable label names — Integration 7A reads them from
    `model.config.id2label` rather than hard-coding.
    """
    set_seed(training_args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Bulletproof CI Compatibility Fix ---
    if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "unwrap_model"):
        orig_unwrap = trainer.accelerator.unwrap_model
        def safe_unwrap(model, *args, **kwargs):
            # Remove problematic kwargs for older accelerate versions in CI
            kwargs.pop("keep_torch_compile", None)
            kwargs.pop("keep_fp32_wrapper", None)
            return orig_unwrap(model, *args, **kwargs)
        trainer.accelerator.unwrap_model = safe_unwrap
    # ----------------------------------------

    trainer.train()
    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Read label names from trainer.model.config.id2label (do not hard-code).

    Returns: {"accuracy": float, "macro_f1": float, "per_class_f1": {label_name: f1, ...}, "per_class_precision": {...}, "per_class_recall": {...}}
    """
    from sklearn.metrics import precision_score, recall_score # Added imports locally

    prediction_output = trainer.predict(tokenized_test)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    id2label = {int(idx): name for idx, name in trainer.model.config.id2label.items()}
    label_ids = sorted(id2label)
    
    per_class_scores = f1_score(
        labels, predictions, average=None, labels=label_ids, zero_division=0
    )
    # Added precision and recall calculation
    per_class_prec_scores = precision_score(
        labels, predictions, average=None, labels=label_ids, zero_division=0
    )
    per_class_rec_scores = recall_score(
        labels, predictions, average=None, labels=label_ids, zero_division=0
    )

    per_class_f1 = {
        id2label[label_id]: float(score)
        for label_id, score in zip(label_ids, per_class_scores)
    }
    # Added mapping for precision and recall
    per_class_precision = {
        id2label[label_id]: float(score)
        for label_id, score in zip(label_ids, per_class_prec_scores)
    }
    per_class_recall = {
        id2label[label_id]: float(score)
        for label_id, score in zip(label_ids, per_class_rec_scores)
    }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision, # Added
        "per_class_recall": per_class_recall,       # Added
    }


def main() -> None:
    """Orchestrate the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    ds = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if os.environ.get("DATA_PATH") is None:
        training_args = make_training_args(output_dir)
    else:
        training_args = make_training_args(output_dir, epochs=4, batch_size=4)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # Save locally (model/ is gitignored)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV
    pred_logits = trainer.predict(tokenized["test"]).predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)
    id2label = trainer.model.config.id2label
    
    # Initialize dictionary for DataFrame
    df_data = {
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    }
    
    # Add per-class probability columns
    for i, label_name in id2label.items():
        df_data[f"prob_{label_name}"] = pred_probs[:, i].tolist()
        
    df_out = pd.DataFrame(df_data)
    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Confusion matrix (for the evaluation report)
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    cm_df = pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values()))
    print(cm_df.to_string())
    
    # Save confusion matrix to CSV
    cm_df.to_csv("confusion_matrix.csv")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()
