# Master PhD Experiment Comparison Table

This document tracks the progression of the PhD research from baseline neural features to the current 3-tier stacking peak.

| Milestone | Configuration | Bal. Acc | Sensitivity | Specificity | Key Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Milestone 1** | Deep Features (Baseline) | 68.2% | 65.1% | 71.3% | ResNet50 Extract |
| **Milestone 2** | LCS Sensitivity Baseline | 69.5% | **84.1%** | 62.4% | Safety Threshold |
| **Milestone 3** | LCS Balanced Consensus | 71.2% | 70.2% | 72.2% | Majority Voting |
| **Milestone 4** | **FARM-LCS (Fuzzy)** | **76.28%** | 74.1% | 78.5% | Fuzzy Membership |
| **Milestone 5** | LCS Evidential (EUQ) | 73.15% | 71.5% | 74.8% | Uncertainty Quant |
| **Milestone 6** | **3-Tier Stacking (Peak)**| **71.88%** | 72.1% | 71.6% | **Isotonic Calib.**|

> [!NOTE]
> **FARM-LCS** remains the highest overall Balanced Accuracy due to its superior handling of boundary cases via Fuzzy Logic, while **Milestone 6 (Stacking)** provides the best probabilistic calibration for clinical use.

---
### Directory Links
- [FARM-LCS](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/FARM_LCS_Fuzzy_Milestone)
- [3-Tier Stacking](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/LCS_Ensemble_Stacking)
- [Evidential (EUQ)](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/LCS_Evidential_Uncertainty)
- [Safety Baseline](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/LCS_Sensitivity_Baseline)
