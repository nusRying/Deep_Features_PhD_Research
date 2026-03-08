# Experiment C: Best Overall Balance (Diagnostic Efficiency)

This experiment represents the **Theoretical Performance Limit** of the current research pipeline, achieving the optimal coordination between safety and accuracy.

## 🎯 Primary Goal

Achieve the highest possible balance between Sensitivity and Specificity across varying datasets.

## 🔬 Technical Approach

- **Core Phase**: Derived from `standalone_v6`.
- **3-Tier Stacking Architecture**:
  - **Tier 1**: 5x parallel ExSTraCS experts.
  - **Tier 2**: Meta-Learner coordination (LogReg + Random Forest) to decide which expert to trust.
  - **Tier 3**: Isotonic Calibration for real-world probability.
- **Noise Removal**: **LCS-Guided Attribute Pruning** removed ~50% of redundant features without loss of signal.

## 📊 Performance Milestone

- **Internal Balanced Accuracy**: **73.3%** (Global Peak)
- **External Balanced Accuracy**: 70.8%
- **Internal Sensitivity**: 81.1%

## 🚀 Reproduction

To reproduce the global peak balanced results:

1. Ensure the `explicit` conda environment is active.
2. Execute the local `main.py`.

```powershell
python main.py
```
