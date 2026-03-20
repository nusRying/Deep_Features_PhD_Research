## 1. The Deep Learning baseline (`dl_baseline`)

The primary control group uses a state-of-the-art EfficientNet encoder for deep feature extraction. While the model achieves high internal validation scores, it fails to generalize to external clinical archives.

| Evaluation Context  | Balanced Accuracy | Sensitivity | Specificity | Overall Accuracy |
| :------------------ | :---------------- | :---------- | :---------- | :--------------- |
| **Internal (ISIC)** | 80.68%            | 73.62%      | 87.74%      | 82.85%           |
| **External (HAM)**  | **50.14%**        | 16.22%      | 84.05%      | 70.82%           |

![Figure: Deep Learning Confusion Matrix (Failure to Generalize)](file:///c:/Users/umair/Videos/PhD/PhD%20Data/Week%205%20Febuary/Deep_Features_Experiment/Main%20Codebase/dl_baseline/results/Plots/cm_external.png)

**Conclusion:** A "Generalization Gap" of **~30%** in Balanced Accuracy proves that raw deep features are highly sensitive to site-specific noise and do not capture universal clinical heuristics.
