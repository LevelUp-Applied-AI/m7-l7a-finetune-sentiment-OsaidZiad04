"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    # 1. Define bin edges and bucket centers
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bucket_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # 2. Extract max probability (confidence) and predicted class for each example
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true)
    
    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins)
    
    # 3. Assign to buckets and compute accuracy
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        # Include the right edge for the last bin
        if i == n_bins - 1:
            mask = (max_probs >= lower) & (max_probs <= upper)
        else:
            mask = (max_probs >= lower) & (max_probs < upper)
            
        count = np.sum(mask)
        bucket_counts[i] = count
        
        if count > 0:
            bucket_accuracies[i] = np.mean(correct[mask])
        else:
            bucket_accuracies[i] = 0.0 # Return 0 for empty buckets to avoid NaN in plots
            
    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true)
    
    N = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        if i == n_bins - 1:
            mask = (max_probs >= lower) & (max_probs <= upper)
        else:
            mask = (max_probs >= lower) & (max_probs < upper)
            
        count = np.sum(mask)
        if count > 0:
            bucket_accuracy = np.mean(correct[mask])
            bucket_confidence = np.mean(max_probs[mask])
            
            # ECE Formula
            weight = count / N
            ece += weight * np.abs(bucket_accuracy - bucket_confidence)
            
    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)