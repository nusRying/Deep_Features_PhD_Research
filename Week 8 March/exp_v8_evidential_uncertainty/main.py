import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# Add Root for Shared Utils and External
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR.parent / "data"
EXTERNAL_DIR = SCRIPT_DIR / "external"

sys.path.append(str(ROOT_DIR))
import shared_utils

sys.path.insert(0, str(EXTERNAL_DIR / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_v8_euq(smoke_test=False):
    logger.info("--- PHASE 8: EVIDENTIAL UNCERTAINTY (EUQ-LCS) ---")
    
    if smoke_test:
        iters = 5000
        n_pop = 500
    else:
        iters = 500000
        n_pop = 5000
        
    # 1. Load Data via Shared Utils
    X, y, feature_names = shared_utils.load_clinical_data(DATA_DIR)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Train Novel EUQ-Experts (Committee of 3)
    logger.info("Training EUQ-Experts...")
    experts = []
    for i in range(3):
        logger.info(f"Expert {i+1}/3 Training...")
        model = ExSTraCS(learning_iterations=iters, N=n_pop, random_state=42+i)
        model.fit(X_train, y_train)
        experts.append(model)
        
    # 3. Perform Evidential Inference (EUQ-LCS)
    logger.info("Extracting Evidential Belief Masses...")
    
    ensemble_evidential = []
    for model in experts:
        # New Hijacked Method: [prob_class_0, prob_class_1, ignorance]
        masses = model.predict_evidential(X_test, K=1.0)
        ensemble_evidential.append(masses)
        
    # Average the belief masses across the committee
    mean_evidential = np.mean(ensemble_evidential, axis=0)
    
    # Extract decision and uncertainty
    # P(Benign), P(Malignant), Ignorance
    prob_B, prob_M, ignorance = mean_evidential[:, 0], mean_evidential[:, 1], mean_evidential[:, 2]
    
    # Final prediction is max between B and M (ignoring the ignorance mass for prediction)
    y_pred = (prob_M > prob_B).astype(int)
    
    # Calculate Standard Metrics
    ba_score = balanced_accuracy_score(y_test, y_pred)
    logger.info(f"Phase 8 EUQ-LCS Balanced Accuracy: {ba_score:.4f}")
    
    # 4. Uncertainty Auditing
    # Track the average uncertainty (ignorance) when the model is correct vs wrong
    correct_mask = (y_pred == y_test)
    wrong_mask = (y_pred != y_test)
    
    fn_mask = (y_pred == 0) & (y_test == 1)
    fp_mask = (y_pred == 1) & (y_test == 0)
    
    avg_uncert_correct = np.mean(ignorance[correct_mask]) if np.any(correct_mask) else 0
    avg_uncert_wrong = np.mean(ignorance[wrong_mask]) if np.any(wrong_mask) else 0
    avg_uncert_fn = np.mean(ignorance[fn_mask]) if np.any(fn_mask) else 0
    
    logger.info(f"Uncertainty when CORRECT: {avg_uncert_correct:.4f}")
    logger.info(f"Uncertainty when ERROR:   {avg_uncert_wrong:.4f}")
    logger.info(f"Uncertainty on FALSE NEGATIVES: {avg_uncert_fn:.4f}")
    
    results = {
        "balanced_accuracy": float(ba_score),
        "uncertainty_correct": float(avg_uncert_correct),
        "uncertainty_wrong": float(avg_uncert_wrong),
        "uncertainty_fn": float(avg_uncert_fn),
        "innovation_status": "Dempster-Shafer Hijack Successful"
    }
    
    with open(SCRIPT_DIR / "results" / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(SCRIPT_DIR / "results" / "uncertainty_report.txt", 'w') as f:
        f.write("EUQ-LCS (Dempster-Shafer) Clinical Uncertainty Report\n")
        f.write("===================================================\n")
        f.write(f"Balanced Accuracy: {ba_score:.4f}\n")
        f.write(f"Average 'Ignorance' Mass on Correct Diagnoses: {avg_uncert_correct:.4f}\n")
        f.write(f"Average 'Ignorance' Mass on Erroneous Diagnoses: {avg_uncert_wrong:.4f}\n")
        f.write(f"Average 'Ignorance' Mass on Missed Melanomas (FN): {avg_uncert_fn:.4f}\n")
        f.write("\nConclusion: If Uncertainty_Wrong > Uncertainty_Correct, the model "
                "successfully 'knows what it doesn't know', proving the Evidential architecture works.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v8_euq(smoke_test=args.smoke_test)
