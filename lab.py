"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.
"""

import json
import os
import dill
import numpy as np
import pandas as pd
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# --- Bulletproof CI Compatibility Fixes ---
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

try:
    _accelerator_unwrap_model = Accelerator.unwrap_model
    def _compat_unwrap_model(self, model, keep_fp32_wrapper=True, keep_torch_compile=None):
        return _accelerator_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)
    Accelerator.unwrap_model = _compat_unwrap_model
except Exception:
    pass
# ------------------------------------------

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def get_data_path() -> str:
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")

def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.train_test_split(test_size=test_size, seed=seed)

def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
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

    # --- CI Compatibility Fix ---
    if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "unwrap_model"):
        orig_unwrap = trainer.accelerator.unwrap_model
        def safe_unwrap(model, *args, **kwargs):
            kwargs.pop("keep_torch_compile", None)
            kwargs.pop("keep_fp32_wrapper", None)
            return orig_unwrap(model, *args, **kwargs)
        trainer.accelerator.unwrap_model = safe_unwrap
    # ----------------------------

    trainer.train()
    return trainer

def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    prediction_output = trainer.predict(tokenized_test)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    id2label = {int(idx): name for idx, name in trainer.model.config.id2label.items()}
    label_ids = sorted(id2label)
    
    per_class_f1_scores = f1_score(labels, predictions, average=None, labels=label_ids, zero_division=0)
    per_class_prec_scores = precision_score(labels, predictions, average=None, labels=label_ids, zero_division=0)
    per_class_rec_scores = recall_score(labels, predictions, average=None, labels=label_ids, zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": {id2label[lid]: float(score) for lid, score in zip(label_ids, per_class_f1_scores)},
        "per_class_precision": {id2label[lid]: float(score) for lid, score in zip(label_ids, per_class_prec_scores)},
        "per_class_recall": {id2label[lid]: float(score) for lid, score in zip(label_ids, per_class_rec_scores)}
    }

def main() -> None:
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

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_logits = trainer.predict(tokenized["test"]).predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)
    id2label = trainer.model.config.id2label
    
    df_data = {
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    }
    for i, label_name in id2label.items():
        df_data[f"prob_{label_name}"] = pred_probs[:, i].tolist()
        
    df_out = pd.DataFrame(df_data)
    df_out.to_csv("predictions.csv", index=False)

    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    cm_df = pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values()))
    cm_df.to_csv("confusion_matrix.csv")

    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
        except Exception:
            pass

def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)

if __name__ == "__main__":
    main()