# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram demonstrates systematic over-confidence. The empirical accuracy bars consistently fall below the dashed "Perfect calibration" line for high-confidence predictions (buckets > 0.5). For example, in the 0.8–0.9 confidence bucket (centered at 0.85), the actual empirical accuracy is noticeably lower, hovering around ~0.70. This means when the model is 85% sure, it is only correct about 70% of the time.

## Expected Calibration Error

**ECE:** 0.1147 (11.47%)

An ECE of 11.47% indicates that the model's predicted probabilities deviate from the true accuracy by approximately 11.5% on average across all bins. For production use, this means the raw softmax probabilities cannot be fully trusted as true likelihoods of correctness. Relying on them directly for critical routing (e.g., auto-flagging negative reviews) will result in higher-than-expected error rates.

## A specific calibration pattern

**Pattern:** Systematic Over-confidence. 
**Reasoning:** This is a well-documented phenomenon in modern neural networks (like DistilBERT). The model is optimized using Cross-Entropy loss, which heavily penalizes uncertainty. To minimize loss during training, the network learns to push logits to extreme values, forcing the softmax outputs closer to 1.0 or 0.0. Because the dataset has overlapping semantic boundaries (e.g., distinguishing "neutral" from slightly "positive"), the model forcefully categorizes them with high confidence rather than expressing the inherent uncertainty of the text.

## A proposed engineering action

**Engineering Action:** Implement **Temperature Scaling** as a post-processing step. By introducing a learned scalar parameter (Temperature, $T > 1$) to divide the logits before the softmax activation, we can "soften" the extreme probabilities. This calibration step preserves the original predictions (the `argmax` remains identical) but directly reduces the ECE, yielding trustworthy confidence scores that can be safely used with static decision thresholds (e.g., `>= 0.90`) in the production pipeline.