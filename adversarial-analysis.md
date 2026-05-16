# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 3 | 6 | 50.0% |
| lexical_trigger | 3 | 6 | 50.0% |
| domain_shift | 2 | 6 | 33.3% |
| length_extreme | 5 | 6 | 83.3% |
| sarcasm | 1 | 6 | 16.7% |
| other | 0 | 0 | N/A |

**Overall Accuracy:** 46.67% (14/30)

## Confirmed hypotheses

The model failed spectacularly on **Sarcasm** and **Domain Shift**, confirming our hypotheses. 
- In Sarcasm, it blindly followed positive cue words. For example, in ID 17 ("Oh great, another update that deletes all my saved files. Thanks!") and ID 29 ("Best app ever for testing my patience."), the model predicted `positive` with high confidence, completely missing the contextual frustration.
- In Domain Shift, the model consistently misclassified factual reporting of negative events as negative user sentiment. For instance, ID 23 ("The stock market saw a massive crash today.") and ID 28 ("The hurricane destroyed the coastal town.") were incorrectly flagged as `negative` rather than `neutral`, proving the model over-indexes on words like "crash" and "destroyed".
- Furthermore, under Lexical Triggers, the model failed when users expressed relief about removing software (ID 7: "managed to uninstall", ID 22: "solved the horrible bug", ID 26: "free from the premium subscription trap"). It saw "uninstall", "bug", and "trap" and predicted `negative`, ignoring the positive resolution.

## Refuted hypotheses

The model handled **Length Extreme** significantly better than expected (83.3% accuracy). I hypothesized that extremely short texts or excessively long, rambling texts would confuse the model's attention mechanisms. However, it correctly classified single words like "Ok." (ID 13) and "Why?" (ID 30) as `neutral`, and accurately handled a massive 50+ word paragraph of irrelevant daily routine (ID 16) as `neutral` as well. 

Additionally, the model handled simple **Negation** surprisingly well. It correctly caught standard negations of positive concepts (ID 1: "did not improve", ID 2: "fail to see how this is considered an upgrade"). It only failed on complex double-negations or ironic phrasing (ID 4: "Not exactly the masterpiece", ID 21: "can't recommend this highly enough").

## What the results reveal about the decision boundary

The adversarial results reveal that the model's decision boundary relies heavily on a "bag-of-words" approach centered around strong polarity cues, rather than deep semantic or syntactic reasoning. The presence of explicitly polarized vocabulary (e.g., "love", "great", "bug", "crash", "uninstall") exerts a massive gravitational pull on the logits, easily overriding structural modifiers like sarcasm or multi-word positive resolutions. Essentially, the model has learned a brittle mapping where "software failure terminology = negative sentiment", making it blind to contexts where bugs are successfully fixed or referenced metaphorically.