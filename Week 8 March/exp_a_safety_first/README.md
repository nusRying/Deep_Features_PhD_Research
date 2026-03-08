# Experiment A: Catching Cancer (Safety-First Policy)

This experiment establishes the laboratory record for **Diagnostic Safety**, ensuring that malignant lesions are prioritized even at the cost of higher false alarms.

## 🎯 Primary Goal

Ensure that malignant lesions are never missed by maximizing the model's **Sensitivity**.

## 🔬 Technical Approach

- **Core Phase**: Derived from `standalone_v2`.
- **Hybrid Probability Calibration**: Combines standard classification with a **Safety Threshold Offset**.
- **The Shift**: Custom probability scaling shifts the decision boundary to be more "suspicious" of benign-looking but potentially harmful features.

## 📊 Performance Milestone

- **Internal Sensitivity**: 82.6%
- **External Sensitivity (HAM10000)**: **72.6%** (Laboratory Record)
- **Balanced Accuracy**: 73.7%

## 🚀 Reproduction

To reproduce these specific safety-first results:

1. Ensure the `explicit` conda environment is active.
2. Execute the local `main.py`.

```powershell
python main.py
```
