# PhD Experiment: EUQ-LCS (Evidential Uncertainty)

## Overview
**EUQ-LCS** (Evidential Uncertainty Quantification) integrates **Dempster-Shafer Theory (DST)** into the Learning Classifier System. It quantifies "Conflict" and "Ignorance" in rule-based decisions, allowing the model to flag cases where it lacks sufficient evidence to make a high-confidence prediction.

## Performance Metrics
| Metric | Internal Validation | External Validation |
| :--- | :---: | :---: |
| **Balanced Accuracy (BA)** | 74.15% | 73.15% |
| **Sensitivity** | 63.20% | 61.20% |
| **Specificity** | 85.10% | 85.10% |

## Implementation Notes
- **Theory**: Evidential Reasoning (Dempster-Shafer Hijack of rule weights).
- **Goal**: Provide a safety layer by measuring evidence mass instead of simple probabilities.
