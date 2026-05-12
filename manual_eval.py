"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    # 1. Prepare model for evaluation and detect its device
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_probs = []

    # 2. Iterate in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 3. Tokenize with required parameters and move to device
        inputs = tokenizer(
            batch_texts, 
            truncation=True, 
            max_length=128, 
            padding=True, 
            return_tensors='pt'
        ).to(device)
        
        # 4. Forward pass without gradient tracking
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 5. Softmax & Argmax
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Move back to CPU and convert to numpy
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # 6. Concatenate all batches into final arrays
    final_preds = np.concatenate(all_preds)
    final_probs = np.vstack(all_probs)
    
    return final_preds, final_probs


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Identify unique classes across both true and predicted arrays
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # Accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    per_class = {}
    f1_scores = []
    
    for c in classes:
        # True Positives, False Positives, False Negatives
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        # Precision (guard divide-by-zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (guard divide-by-zero)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[int(c)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        f1_scores.append(f1)
        
    # Macro F1 (mean of per-class f1 scores)
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }