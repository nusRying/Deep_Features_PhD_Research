# Phase 4: Stacking Ensemble & Meta-Learning

## Overview

This phase introduces the **Stacking Ensemble** architecture. Instead of a single model, we train a committee of 3 ExSTraCS experts and use a Logistic Regression meta-learner to adjudicate their votes.

## Key Goals

- Overcome the variance of single-model initialization.
- Implement "Out-of-Fold" logic to train the meta-learner on unseen validation data.

## Results Summary

- **Performance**: Significant reduction in variance.
- **Contribution**: Introduction of the "Tier 2" adjudicator which became the foundation for future accuracy gains.

## Reproduction

```powershell
python main.py
```
