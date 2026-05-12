import sys
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Adding the stretch directory to Python's path so it can find your files
sys.path.append('stretch')
sys.path.append('stretch/tuesday')

from manual_eval import manual_predict
from calibration import reliability_diagram, expected_calibration_error, plot_reliability

def main():
    print("Loading model and tokenizer...")
    model_dir = "model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("Loading test data...")
    # Using the predictions.csv we generated earlier to get the true labels easily
    df = pd.read_csv("predictions.csv")
    texts = df['text'].tolist()
    
    # Map string labels back to integers based on your ID2LABEL
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    y_true = np.array([label2id[lbl] for lbl in df['label']])

    print("Running manual inference (this might take a few minutes)...")
    preds, probs = manual_predict(model, tokenizer, texts, batch_size=8)

    print("\nCalculating Calibration Metrics...")
    # Calculate ECE
    ece = expected_calibration_error(probs, y_true, n_bins=10)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Generate Reliability Diagram
    centers, accs, counts = reliability_diagram(probs, y_true, n_bins=10)
    plot_reliability(centers, accs, counts, "reliability_diagram.png")
    print("Saved 'reliability_diagram.png'")

if __name__ == "__main__":
    main()