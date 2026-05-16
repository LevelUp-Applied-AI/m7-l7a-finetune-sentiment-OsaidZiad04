"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    df = pd.read_csv(adv_csv_path)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    predicted_labels = []
    predicted_probs = []
    correct_flags = []
    
    for _, row in df.iterrows():
        text = str(row['text'])
        expected_label = str(row['expected_label']).strip().lower()
        
        inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0, pred_idx].item()
        
        pred_label = model.config.id2label[pred_idx]
        
        predicted_labels.append(pred_label)
        predicted_probs.append(pred_prob)
        # Check if prediction matches expected label
        correct_flags.append(pred_label.lower() == expected_label)
        
    df['predicted_label'] = predicted_labels
    df['predicted_probability'] = predicted_probs
    df['correct'] = correct_flags
    
    return df


def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()