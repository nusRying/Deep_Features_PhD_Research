# Experiment B: Avoiding False Alarms (Conservative Policy)

This experiment establishes the laboratory record for **Diagnostic Precision**, minimizing unnecessary clinical procedures and patient anxiety.

## 🎯 Primary Goal

Minimize unnecessary biopsies by maximizing the model's **Specificity**.

## 🔬 Technical Approach

- **Core Phase**: Derived from `standalone_v3`.
- **Democratic Ensemble**: Utilizes a consensus-voting mechanism where a diagnosis is only confirmed if a strong majority of experts agree.
- **Voter Pool**: 5 Multi-seed ExSTraCS models trained with rigorous confidence requirements.

## 📊 Performance Milestone

- **Internal Specificity**: **87.2%** (Laboratory Record)
- **Balanced Accuracy**: 71.3%

## 🚀 Reproduction

To reproduce these conservative consensus results:

1. Ensure the `explicit` conda environment is active.
2. Execute the local `main.py`.

```powershell
python main.py
```
