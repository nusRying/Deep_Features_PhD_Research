# Phase 5: SMOTE-ENN & Universal Calibration

## Overview

This phase focused on **Class Imbalance** and **Probability Calibration**. It integrates SMOTE-ENN for cleaning decision boundaries and prepares the pipeline for clinical reliability.

## Key Goals

- Apply SMOTE-ENN balancing at each fold level.
- Calibrate the model output to ensure "Confidence" maps to "Probability."

## Results Summary

- **Impact**: Dramatically improved sensitivity for malignant lesions without collapsing specificity.
- **Contribution**: The "Data Engineering" layer (L2) was finalized here.

## Reproduction

```powershell
python main.py
```
