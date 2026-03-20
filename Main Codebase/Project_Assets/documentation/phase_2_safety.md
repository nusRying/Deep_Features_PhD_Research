# PhD Phase 2: Safety - The Clinical Baseline

## Overview & Motivation
The primary objective of Phase 2 was to establish a rigorous clinical baseline for safety prediction. In clinical practice, decision-making often relies on discrete tabular data (age, symptoms, lab results). By using a Learning Classifier System (LCS), we aimed to discover interpretable "if-then" rules that mirror clinical logic, providing a foundation against which subsequent deep-learning-enhanced models could be measured.

## Methodology & Implementation
- **Feature Set**: 14 Clinical Features + 4 Pathological Features (18 total).
- **LCS Engine**: `skExSTraCS` v2.1 (Supervised Learning Classifier System).
- **Training**: 100,000 iterations for convergence.
- **Rule Discovery**: Utilized the "Center-Spread" interval matching system for handling numerical ranges.

### Mathematical Foundation: Boolean Interval Matching
In `skExSTraCS`, an input feature $x_i$ matches a rule if it falls within the rule's defined interval $[c_i - s_i, c_i + s_i]$.
$$M(x_i, c_i, s_i) = \begin{cases} 1 & \text{if } c_i - s_i \le x_i \le c_i + s_i \\ 0 & \text{otherwise} \end{cases}$$
The rule utility (fitness $\Omega$) is calculated based on accuracy and sparsity, favoring rules that cover more samples with minimal error.

## Expectations (Hypothesis)
We hypothesized that the clinical baseline would provide a stable, "safe" prediction model with high sensitivity but lower specificity, as clinical indicators are often reactive rather than proactive in certain diagnoses.

## Results: What We Got
| Metric | Internal Validation | External Validation |
| :--- | :---: | :---: |
| **Balanced Accuracy (BA)** | 72.82% | 71.79% |
| **Sensitivity** | 84.10% | 82.59% |
| **Specificity** | 61.54% | 61.00% |

### Analysis
The results confirmed our hypothesis: high sensitivity (84.10%) indicates the model is excellent at identifying positive safety signals, but the lower specificity (61.54%) suggests a high false-alarm rate when relying solely on clinical data. This gap motivated the transition to visual feature integration in Phase 4.
