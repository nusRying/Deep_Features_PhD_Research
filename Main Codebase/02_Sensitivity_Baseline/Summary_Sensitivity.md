# PhD Experiment: LCS Safety-First Configuration

## Overview
The **Safety-First** configuration focuses on maximizing **Sensitivity** (Recall) to ensure that the majority of malignant cases are correctly identified, even at the cost of higher false positives (lower specificity). This is critical for screening applications where missing a case is more dangerous than an unnecessary follow-up.

## Performance Metrics
| Metric | Internal Validation | External Validation |
| :--- | :---: | :---: |
| **Balanced Accuracy (BA)** | 72.82% | 71.79% |
| **Sensitivity** | **84.10%** | **82.59%** |
| **Specificity** | 61.54% | 61.00% |

## Implementation Notes
- **LCS Engine**: skExSTraCS with lowered decision threshold ($\approx 0.20$).
- **Goal**: High-recall baseline for comparison against balanced and discovery-oriented models.
