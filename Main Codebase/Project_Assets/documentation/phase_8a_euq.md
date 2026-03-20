# PhD Phase 8a: EUQ-LCS - Evidential Uncertainty Quantification

## Overview & Motivation
Phase 8a addressed the "blind trust" issue in clinical AI. In previous phases, the model always provided a prediction, even when the input data was outside the known distribution. We introduced **Evidential Deep Learning (EDL)** to enable the LCS engine to "know when it doesn't know." By quantifying **Epistemic Uncertainty**, we could implement an abstinence mechanism, only allowing high-confidence signals to proceed to a final clinical decision.

## Methodology & Implementation
- **Base Engine**: 5-expert Consensus from Phase 6.
- **Module**: Evidential Deep Learning (EDL).
- **Abstinence Rule**: Uncertainty Score $u < 0.15$ threshold.
- **Goal**: Recovery of Accuracy through reliable filtering.

### Mathematical Foundation: Dirichlet Distribution & Evidence
We model the prediction as a **Dirichlet Distribution** $Dir(\mathbf{p} ; \boldsymbol{\alpha})$, where $\boldsymbol{\alpha} = [ \alpha_1, \dots, \alpha_K ]$ represents the evidence for each class.
For each class $k$, the evidence $e_k$ is accrued from the LCS rule matches:
$$\alpha_k = e_k + 1$$
The **Total Strength ($S$)** and **Uncertainty ($u$)** are then:
$$S = \sum_{k=1}^{K} \alpha_k$$
$$u = \frac{K}{S}$$
where $K$ is the number of classes. When $u \ge 0.15$, the model abstains from predicting, identifying the case as "Unreliable."

## Expectations (Hypothesis)
We expected that by filtering out high-uncertainty samples, the **Balanced Accuracy** and **Sensitivity** would recover on the remaining high-confidence set.

## Results: What We Got
| Metric | Internal Validation | External Validation |
| :--- | :---: | :---: |
| **Balanced Accuracy (BA)** | 74.15% | 73.15% |
| **Sensitivity** | 63.20% | 61.20% |
| **Specificity** | 85.10% | 85.10% |

### Analysis
The EUQ-LCS successfully recovered the **External BA to 73.15%** (higher than Phases 4 and 6) and improved Sensitivity to 61.20% while maintaining high Specificity. This proved that quantifying uncertainty is key to stability, but the final performance bottleneck remained the "step-function" nature of the boolean intervals—leading to the Phase 8b Fuzzy breakthrough.
