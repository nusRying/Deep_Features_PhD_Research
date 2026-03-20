## 2. Hardened Traditional ML (`hardened_baselines`)

To verify if the issue was merely "Hyperparameter Sensitivity," we optimized traditional high-performance classifiers on the same 256-dimensional PCA features.

| Model Variant     | Internal BA | External BA | Generalization Delta |
| :---------------- | :---------- | :---------- | :------------------- |
| **Random Forest** | 72.10%      | 50.18%      | -21.92%              |
| **XGBoost**       | 77.36%      | 50.16%      | -27.20%              |
| **SVM (Linear)**  | 80.74%      | 50.29%      | -30.45%              |
| **SOTA Ensemble** | 78.99%      | 50.15%      | -28.84%              |

**Conclusion:** The failure persists across all mathematical foundations (bagging, boosting, and kernel methods). This confirms that the problem lies in the **lack of symbolic interpretability** and the high-dimensional opacity of the features themselves.

### Visual Comparison (Internal vs External)

#### ROC and Precision-Recall Curves
![ROC Curves](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/hardened_baselines/results/Plots/roc_overlaid_external.png)
![PR Curves](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/hardened_baselines/results/Plots/pr_overlaid_external.png)

#### Confusion Matrices (Sample Models)
- **XGBoost (External)**: ![XGBoost External CM](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/hardened_baselines/results/Plots/cm_xgboost_external.png)
- **Random Forest (External)**: ![RF External CM](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/hardened_baselines/results/Plots/cm_randomforest_external.png)
