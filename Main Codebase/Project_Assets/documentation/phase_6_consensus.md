# PhD Phase 6: Consensus - The Expert Ensemble Approach

## Overview & Motivation
Phase 6 focused on solving the variance issue observed in Phase 4. While specific, a single LCS learner can be prone to the "curse of dimensionality" when handling 2000+ features. To address this, we developed a Parallel Consensus Architecture. By allowing five independent skExSTraCS "experts" to converge separately on the same data, we could filter out noisy rules and rely on a majority-driven "Clinical Consensus."

## Methodology & Implementation
- **Experts**: 5x `skExSTraCS` models initialized with different random seeds.
- **Iterations**: 500,000 max iterations per expert (Deep Convergence).
- **Mechanism**: Synchronous majority voting on the final test set.

### Mathematical Foundation: Majority Voting & Mode Rule
The consensus prediction $Y_{cons}$ is derived from the mode of the individual expert outputs $y_1, y_2, \dots, y_5$:
$$V = \{y_1, y_2, y_3, y_4, y_5\}$$
$$Y_{cons} = \text{mode}(V)$$
To calculate the **Mean Rule Probability ($\bar{P}$)** for clinical utility:
$$\bar{P} = \frac{1}{k} \sum_{i=1}^{k} \Phi(x_i)$$
where $\Phi(x_i)$ is the local accuracy of the matching rule in expert $i$. This ensures the model only fires when the most accurate rules from each expert align.

## Expectations (Hypothesis)
This phase aimed to stabilize the **Specificity** (above 85%) while attempting to recover some of the **Sensitivity** lost in Phase 4 through the "Wisdom of the Crowd" effect.

## Results: What We Got
| Metric | Internal Validation | External Validation |
| :--- | :---: | :---: |
| **Balanced Accuracy (BA)** | 73.22% | 71.29% |
| **Sensitivity** | 59.20% | 55.34% |
| **Specificity** | 87.20% | 87.24% |

### Analysis
Phase 6 achieved its goal of maximum **External Specificity (87.24%)**. However, External Sensitivity remained low (55.34%), confirming that an ensemble of *certain* experts might be too restrictive. This identified the need for "Uncertainty Quantification" to differentiate between a truly negative case and one where the experts simply don't have enough "evidence" to decide—leading to Phase 8a.
