# Phase 8: Evidential Uncertainty Quantification (EUQ-LCS)

**Methodological Innovation for PhD Thesis**

## The Breaking Point

Traditional ExSTraCS uses a standard probability estimation vector (e.g., $P(\text{Malignant}) + P(\text{Benign}) = 1.0$). In oncology, this "forced choice" paradigm is inherently flawed. If the model sees a lesion that lies completely outside of its training manifold, it will arbitrarily assign probabilities that sum to 1.0, masking its ignorance.

## The Innovation

We performed an **Architectural Hijacking** of the `skExSTraCS` `Prediction.py` module. We integrated **Dempster-Shafer Belief Functions** to quantify the model's epistemic uncertainty.

The `getEvidentialProbabilities()` method now outputs a triplet of Belief Masses based on rule support:
`[Mass(Benign), Mass(Malignant), Mass(Theta)]`

Where $\Theta$ represents the model's fundamental _clinical ignorance_ or _uncertainty_.

## Clinical Validation Goal

Our hypothesis is that the model's Ignorance Mass parameter ($\Theta$) will be statistically correlated with model error—specifically, it should spike drastically on **False Negatives** (missed melanomas), proving that the model "knows it doesn't know" before making a fatal mistake.

## File Modifications

- `external/scikit-ExSTraCS-master/skExSTraCS/ExSTraCS.py` -> Injected `predict_evidential()`.
- `external/scikit-ExSTraCS-master/skExSTraCS/Prediction.py` -> Injected Dempster-Shafer theory logic into `getEvidentialProbabilities()`.
- `main.py` -> Central orchestrator for the EUQ-LCS ensemble.

## How to Reproduce

To execute the full 500,000 learning iteration run with the 3 expert EUQ committee:

```powershell
cd "C:\Users\umair\Videos\PhD\PhD Data\Week 5 Febuary\Deep_Features_Experiment\Week 8 March\exp_v8_evidential_uncertainty"
python main.py
```

_Note: For a quick verification test (5,000 iterations), run `python main.py --smoke-test`._
