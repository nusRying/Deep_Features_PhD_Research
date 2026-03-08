import sys
from pathlib import Path
import json
import logging
import warnings
import numpy as np

# Suppress sklearn warnings for clean output
warnings.filterwarnings("ignore")

# Configure Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"

# Direct the script to use the hijacked local library instead of pip installed packages
# This ensures that our modifications to Classifier.match() are utilized!
sys.path.insert(0, str(SCRIPT_DIR / "external" / "scikit-ExSTraCS-master"))
sys.path.insert(0, str(ROOT_DIR))

from skExSTraCS import ExSTraCS
from shared_utils import load_clinical_data
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from shared_utils import load_clinical_data, compute_phd_metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def generate_diagnostic_plots(y_true, y_pred, y_probs, title_prefix, results_dir):
    """Generate CM, ROC, and PR plots for PhD Dissertation."""
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{title_prefix}: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / f"{title_prefix.lower()}_cm.png")
    plt.close()

    # 2. ROC & PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_title(f"{title_prefix}: ROC Curve")
    ax1.legend(loc="lower right")

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)
    ax2.plot(rec, prec, color='green', lw=2, label=f'PR (AP = {ap_score:.2f})')
    ax2.set_title(f"{title_prefix}: Precision-Recall")
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(results_dir / f"{title_prefix.lower()}_curves.png")
    plt.close()

def evaluate_fuzzy_lcs(smoke_test=False):
    logger.info("--- PHASE 8b: FUZZY RULE MORPHING (FARM-LCS) ---")
    
    RESULTS_DIR = SCRIPT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load standardized data
    X, y, feature_names = load_clinical_data(DATA_DIR)
    
    # Split Data manually
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Set the ExSTraCS hyperparameters for PROPER training
    ITERATIONS = 500 if smoke_test else 500000 
    POPULATION = 5000
    N_EXPERTS = 5 
    
    logger.info(f"Training FARM-Experts: {ITERATIONS} iters, {POPULATION} pop, {N_EXPERTS} experts...")
    experts = []
    for i in range(N_EXPERTS):
        logger.info(f"Fuzzy-Expert {i+1}/{N_EXPERTS} Training...")
        model = ExSTraCS(learning_iterations=ITERATIONS, N=POPULATION, random_state=42 + i)
        model.fit(X_train, y_train)
        experts.append(model)
        
    def get_ensemble_audit(data):
        all_preds = []
        all_probs = []
        for model in experts:
            all_preds.append(model.predict(data))
            all_probs.append(model.predict_proba(data)[:, 1]) # Column 1 is Malignant
            
        from scipy.stats import mode
        final_preds, _ = mode(all_preds, axis=0)
        final_probs = np.mean(all_probs, axis=0)
        return final_preds.ravel(), final_probs.ravel()

    logger.info("Calculating Final Audited Results...")
    
    # Internal Performance
    y_pred_train, y_prob_train = get_ensemble_audit(X_train)
    int_metrics = compute_phd_metrics(y_train, y_pred_train, y_prob_train)
    generate_diagnostic_plots(y_train, y_pred_train, y_prob_train, "Internal", RESULTS_DIR)
    
    # External Performance
    y_pred_test, y_prob_test = get_ensemble_audit(X_test)
    ext_metrics = compute_phd_metrics(y_test, y_pred_test, y_prob_test)
    generate_diagnostic_plots(y_test, y_pred_test, y_prob_test, "External", RESULTS_DIR)
    
    logger.info(f"[INTERNAL] BA: {int_metrics['ba']:.4f}, Sens: {int_metrics['sens']:.4f}, Spec: {int_metrics['spec']:.4f}")
    logger.info(f"[EXTERNAL] BA: {ext_metrics['ba']:.4f}, Sens: {ext_metrics['sens']:.4f}, Spec: {ext_metrics['spec']:.4f}")
    
    results = {
        "config": {"iterations": ITERATIONS, "population": POPULATION, "experts": N_EXPERTS},
        "internal": int_metrics,
        "external": ext_metrics,
        "innovation": "Self-Adaptive Fuzzy Rule Morphing (FARM-LCS)"
    }
    
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(RESULTS_DIR / "thesis_report.txt", 'w') as f:
        f.write("PhD THESIS DISCOVERY: FARM-LCS AUDIT\n")
        f.write("=====================================\n")
        f.write("INTERNAL (Train):\n")
        for k, v in int_metrics.items(): f.write(f"  {k.upper()}: {v:.4f}\n")
        f.write("\nEXTERNAL (Generalized Test):\n")
        for k, v in ext_metrics.items(): f.write(f"  {k.upper()}: {v:.4f}\n")
    
    logger.info(f"Report and Plots saved to {RESULTS_DIR}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run with tiny iterations for testing")
    args = parser.parse_args()
    
    if args.smoke_test:
        logger.info("Running Smoke Test...")
        evaluate_fuzzy_lcs(smoke_test=True)
    else:
        evaluate_fuzzy_lcs(smoke_test=False)
