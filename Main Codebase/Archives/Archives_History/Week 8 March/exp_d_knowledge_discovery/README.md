# Experiment D: Knowledge Discovery (Interpretability Layer)

This experiment represents the **Final Technical Frontier** of the PhD research. It moves beyond pure performance to achieve **Explainable Clinical Diagnostics**.

## 🎯 Primary Goal

Harvest the linguistic "IF-THEN" rules from the ensemble of experts to explain _why_ a particular lesion is classified as malignant.

## 🔬 Technical Approach

- **Core Phase**: Derived from `phase_v7_knowledge_discovery`.
- **Knowledge Harvesting**: Traverses the population of all 5 Tier-1 experts to extract high-fitness rules.
- **Consensus Reporting**: Generates a **Clinical Consensus Report** where rules verified by multiple experts are prioritized.
- **Statistical Rigor**: Includes **Bootstrap Confidence Intervals** for all metrics (MCC, Brier Score, ECE).

## 📊 Evaluation Outputs

- **`clinical_consensus_report.txt`**: A human-readable diagnostic guide.
- **`results_v7_knowledge.json`**: Comprehensive statistical matrix for thesis "Results" chapter.

## 🚀 Execution

To generate the final PhD ruleset and report:

1. Ensure the `explicit` conda environment is active.
2. Execute the local `main.py`.

```powershell
python main.py
```

> [!TIP]
> This is a high-iteration run (500k cycles). For a quick verification, use `python main.py --smoke-test`.
