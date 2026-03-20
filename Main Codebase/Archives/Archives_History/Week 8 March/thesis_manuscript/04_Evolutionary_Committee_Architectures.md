# Chapter 4: Evolutionary Committee Architectures

**Aim**: Document the journey from single model to complex ensemble (Matches Phases 2 through 6).

## 4.1 The LCS Baseline (Phase 1)
[Draft Content Here]

## 4.2 Imposing Clinical Safety via Asymmetric Weights (Phase 2 & 3)
[Draft Content Here]

## 4.3 Meta-Learning and The Logistic Adjudicator (Phase 4)
[Draft Content Here]

## 4.4 SMOTE-ENN and Isotonic Probability Calibration (Phase 5 & 6)
[Draft Content Here]

## 4.5 Generalization Results (Closing the Gap)

### Performance Matrix: The Evolutionary Trajectory

| Model Phase                            |   Balanced Accuracy |   Sensitivity (Recall) |   Specificity |   F1-Score |   MCC |
|:---------------------------------------|--------------------:|-----------------------:|--------------:|-----------:|------:|
| Deep Learning Baseline                 |              0.5014 |                 0.1622 |        0.8406 |          0 |     0 |
| Phase 1: Basic LCS                     |              0.7102 |                 0.544  |        0.8765 |          0 |     0 |
| Phase 2 & 3: Clinical Safety Weighting |              0.7179 |                 0.8259 |        0.6099 |          0 |     0 |
| Phase 4: Stacking Ensemble             |              0.7188 |                 0.5868 |        0.8509 |          0 |     0 |
| Phase 6: Global Peak                   |              0.7129 |                 0.5534 |        0.8724 |          0 |     0 |

*(Insert `../thesis_figures/fig_1_performance_evolution.png` here)*
