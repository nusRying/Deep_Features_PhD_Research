# PhD Experiment: LCS Stacking Ensemble

## Overview
The **Stacking** experiment utilizes a heterogeneous ensemble of LCS models. Multiple "base" models are trained, and their predictions are used as features for a secondary "meta-learner" (Stacked Classifier) to improve generalization and robustly handle deep feature complexity.

## Performance Metrics
| Metric | Internal Validation | External Validation |
| :--- | :---: | :---: |
| **Balanced Accuracy (BA)** | 73.30% | 71.88% |
| **Sensitivity** | 71.50% | 58.68% |
| **Specificity** | 75.10% | 85.09% |

## Implementation Notes
- **LCS Engine**: Ensembles of skExSTraCS.
- **Goal**: Investigate if meta-learning can mitigate "noisy" deep feature variances.
