# PhD Phase 4: Stacking - Hybrid Deep Learning Integration

## Overview & Motivation
Phase 4 represents a significant architectural shift: the transition from purely clinical tabular data to a hybrid "deep feature" model. By integrating high-dimensional visual features from a ResNet-50 backbone with the baseline clinical indicators, we aimed to capture pathological patterns invisible to the naked eye. To manage this 2066-dimensional feature space, we implemented a heterogeneous ensemble with a stacked generalization layer.

## Methodology & Implementation
- **Feature Set**: ResNet-50 (2048-dim) + Clinical Baseline (18-dim) = 2066 features.
- **Ensemble Layer**:
    1.  **ExSTraCS (LCS)**: Rule-based discovery.
    2.  **Random Forest**: Non-linear feature interactions.
    3.  **XGBoost**: Gradient-boosted error correction.
- **Meta-Learner**: Logistic Regression (L2 Penalty).

### Mathematical Foundation: Stacked Generalization
The final prediction $\hat{y}$ is a weighted combination of the base learners' probabilities ($p_1, p_2, p_3$), optimized via a Logistic Regression meta-classifier:
$$z = \sum_{i=1}^{n} w_i p_i + b$$
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $w_i$ are the learned weights for each model's contribution. This ensures that if one model (e.g., XGBoost) overfits onto the visual features, the meta-learner can down-weight its influence in favor of the more stable LCS rules.

## Expectations (Hypothesis)
Integration of visual features was expected to drastically increase Specificity (reducing false positives) by identifying concrete pathological markers in the Deep Feature matrix.

## Results: What We Got
| Metric | Internal Validation | External Validation |
| :--- | :---: | :---: |
| **Balanced Accuracy (BA)** | 73.30% | 71.88% |
| **Sensitivity** | 71.50% | 58.68% |
| **Specificity** | 75.10% | 85.09% |

### Analysis
As expected, **External Specificity skyrocketed to 85.09%** (compared to 61% in Phase 2). However, there was a noticeable drop in External Sensitivity (58.68%), suggesting that while the visual features are highly specific, the model's reliance on specific deep features may cause it to miss some broader clinical cases. This trade-off led to the Phase 6 Consensus approach.
