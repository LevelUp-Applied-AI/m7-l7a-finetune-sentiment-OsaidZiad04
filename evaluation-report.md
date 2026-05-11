# Module 7 Week A Lab Evaluation Report

## Dataset
The model was fine-tuned on a curated subset of the AARSynth app reviews dataset. The data was split into a training set (~7,472 examples) and an evaluation set (1,495 examples) across 3 sentiment classes: negative, neutral, and positive.

## Model and hyperparameters
- **Backbone:** `distilbert-base-uncased`
- **Number of labels:** 3
- **Hyperparameters:** Learning rate = 5e-5, Epochs = 2, Batch size = 8, Max length = 128, Seed = 42.
- **Training time:** Approximately 24 minutes on a local CPU.

## Metrics on the test split
### Aggregate:
| Metric | Value |
|---|---|
| Accuracy | 0.6328 |
| Macro-F1 | 0.6324 |

### Per class:
| Class | F1 | Precision | Recall |
|---|---|---|---|
| Positive | 0.6914 | 0.7210 | 0.6642 |
| Neutral | 0.4894 | 0.4601 | 0.5227 |
| Negative | 0.7165 | 0.7322 | 0.7014 |

## Confusion matrix
| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| **Actual Negative** | 350 | 133 | 16 |
| **Actual Neutral** | 100 | 242 | 121 |
| **Actual Positive** | 28 | 151 | 354 |

**Interpretation:** The model is highly effective at distinguishing polar extremes (negative vs. positive). Its primary failure mode is confusing actual polar sentiment (both positive and negative) with neutrality, and vice versa. This is reflected in the low F1 score (0.4894) for the Neutral class compared to the polar classes (~0.70+).

## Three qualitative error examples (one per class)

**1. Actual: Positive, Predicted: Neutral**
- **Original sentence:** "its a good app but we dont have a night mode option for reading in which the background becomes black and text becomes white in color."
- **Predicted probability for the gold label (Positive):** 0.2303
- **Analysis:** The sentence starts with a clear positive sentiment ("good app") but spends the majority of the text describing a missing technical feature in a factual tone. The model likely weighed the long, factual description heavily, averaging the overall sentiment out to neutral.

**2. Actual: Negative, Predicted: Neutral**
- **Original sentence:** "a lot of a bug here"
- **Predicted probability for the gold label (Negative):** 0.4376
- **Analysis:** While "bug" strongly implies a negative experience in the context of apps, the sentence is very short and lacks strong emotional adjectives (like "terrible" or "awful"). The model treated it as a dry, factual statement rather than a negative complaint.

**3. Actual: Neutral, Predicted: Positive**
- **Original sentence:** "when i was being rained on it said no rain in your <url> it is still a good weather app."
- **Predicted probability for the gold label (Neutral):** 0.4531
- **Analysis:** The user points out a factual error made by the app (which is neutral/negative) but concludes with the phrase "still a good weather app." The presence of the strong positive keyword "good" at the very end overpowered the initial factual critique, causing the model to misclassify it as completely positive.

## Hugging Face Hub model URL
https://huggingface.co/Oss04/m7-app-review-sentiment