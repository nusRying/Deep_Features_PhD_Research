import sys
import json
import numpy as np
import pandas as pd
import pickle
import time
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, accuracy_score, 
    recall_score, precision_score, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from project_paths import (
    EXSTRACS_LIB_DIR,
    HYBRID_ISIC_FILE,
    ISIC_METADATA_FILE,
    HYBRID_HAM_FILE,
    HAM_METADATA_FILE,
    RESULTS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
)

# Setup Paths
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

# Output Files
RESULTS_FILE = RESULTS_DIR / "results_hybrid_phd_final.json"
MODEL_FILE = MODELS_DIR / "hybrid_ensemble_phd_final.pkl"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = MODELS_DIR / "checkpoints_opt"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline_execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 789, 2024, 99]
LEARNING_ITERATIONS = 500000 
N_POP = 5000  # Increased for better specificity/sensitivity
BOOTSTRAP_SAMPLES = 1000

def get_metrics_with_ci(y_true, y_prob, threshold=0.5):
    """Calculates metrics and 95% Confidence Intervals via Bootstrapping."""
    def calc_metrics(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        return {
            "ba": balanced_accuracy_score(y_t, y_pred),
            "sens": recall_score(y_t, y_pred),
            "spec": recall_score(y_t, y_pred, pos_label=0),
            "prec": precision_score(y_t, y_pred, zero_division=0),
            "acc": accuracy_score(y_t, y_pred)
        }

    base_metrics = calc_metrics(y_true, y_prob)
    
    # Bootstrapping
    boot_stats = {k: [] for k in base_metrics.keys()}
    logger.info(f"Running {BOOTSTRAP_SAMPLES} bootstrap resamples for CIs...")
    for i in range(BOOTSTRAP_SAMPLES):
        if (i + 1) % 100 == 0:
            logger.info(f"  Bootstrap progress: {i + 1}/{BOOTSTRAP_SAMPLES}")
        indices = resample(np.arange(len(y_true)), replace=True)
        res = calc_metrics(y_true[indices], y_prob[indices])
        for k in boot_stats.keys():
            boot_stats[k].append(res[k])
            
    final_results = {}
    for k in base_metrics.keys():
        lower = np.percentile(boot_stats[k], 2.5)
        upper = np.percentile(boot_stats[k], 97.5)
        final_results[k] = {
            "val": float(base_metrics[k]),
            "ci_lower": float(lower),
            "ci_upper": float(upper)
        }
        
    # Standard CM (not bootstrapped)
    y_pred_base = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_base).ravel()
    final_results["cm"] = [[int(tn), int(fp)], [int(fn), int(tp)]]
    
    return final_results

def plot_diagnostics(y_test, y_prob_test, y_ext, y_prob_ext, suffix=""):
    plt.figure(figsize=(15, 6))
    
    # 1. ROC Curve
    plt.subplot(1, 2, 1)
    for y_t, y_p, label in [(y_test, y_prob_test, "Internal Test"), (y_ext, y_prob_ext, "External HAM")]:
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    # 2. Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for y_t, y_p, label in [(y_test, y_prob_test, "Internal Test"), (y_ext, y_prob_ext, "External HAM")]:
        prec, rec, _ = precision_recall_curve(y_t, y_p)
        plt.plot(rec, prec, label=f'{label}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"diagnostics_{suffix}.png", dpi=300)
    plt.close()

def run_phd_pipeline(smoke_test=False):
    global BOOTSTRAP_SAMPLES, LEARNING_ITERATIONS
    if smoke_test:
        logger.info("SMOKE TEST MODE ENABLED")
        BOOTSTRAP_SAMPLES = 10
        LEARNING_ITERATIONS = 5000
    
    logger.info("--- PHD GOLD STANDARD: OPTIMIZED STACKING ENSEMBLE + SMOTE (v5) ---")
    
    # Data Loading
    logger.info("Loading data...")
    df_feats = pd.read_csv(HYBRID_ISIC_FILE)
    df_meta = pd.read_csv(ISIC_METADATA_FILE)
    df = pd.merge(df_feats, df_meta, on='image_id', how='inner')
    df = df.drop(columns=['lesion_id', 'dx', 'image', 'anatom_site_general', 'localization'], errors='ignore')
    
    deep_cols = [f"deep_{i}" for i in range(2048)]
    other_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    hand_meta_cols = [c for c in other_numeric if c not in deep_cols and c != 'label']
    
    X_deep = df[deep_cols].values
    y = df['label'].values
    
    # PCA & Scaling
    logger.info("Fitting PCA and Scaler...")
    pca = PCA(n_components=256, random_state=42)
    X_deep_pca = pca.fit_transform(X_deep)
    X = np.hstack([X_deep_pca, df[hand_meta_cols].values])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # External Data
    logger.info("Loading external HAM data...")
    df_ham = pd.merge(pd.read_csv(HYBRID_HAM_FILE), pd.read_csv(HAM_METADATA_FILE), on='image_id', how='inner')
    for col in hand_meta_cols:
        if col not in df_ham.columns: df_ham[col] = 0
    X_ham = scaler.transform(np.hstack([pca.transform(df_ham[deep_cols].values), df_ham[hand_meta_cols].values]))
    y_ham = df_ham['label'].values
    
    # ENSEMBLE TRAINING
    models_ensemble = []
    val_probs_all, test_probs_all, ext_probs_all = [], [], []
    
    for i, seed in enumerate(SEEDS):
        ckpt_path = CHECKPOINT_DIR / f"expert_{i}_seed_{seed}.pkl"
        
        if ckpt_path.exists():
            logger.info(f"Resuming Expert {i+1}/5 from checkpoint (Seed: {seed})...")
            with open(ckpt_path, 'rb') as f:
                model = pickle.load(f)
        else:
            logger.info(f"\nTraining Expert {i+1}/5 (Seed: {seed}) with SMOTE...")
            smote = SMOTE(random_state=seed)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            
            model = ExSTraCS(learning_iterations=LEARNING_ITERATIONS, N=N_POP, nu=5) # Increased nu for precision
            model.fit(X_res, y_res)
            
            # Save checkpoint
            with open(ckpt_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Expert {i+1} saved to {ckpt_path.name}")
            
        models_ensemble.append(model)
        val_probs_all.append(model.predict_proba(X_val)[:, 1])
        test_probs_all.append(model.predict_proba(X_test)[:, 1])
        ext_probs_all.append(model.predict_proba(X_ham)[:, 1])
        
    p_val = np.mean(val_probs_all, axis=0)
    p_test = np.mean(test_probs_all, axis=0)
    p_ext = np.mean(ext_probs_all, axis=0)
    
    # META-LEARNER (STACKING)
    logger.info("Training Meta-Learner (Stacking Ensemble)...")
    X_val_meta = np.column_stack(val_probs_all)
    meta_learner = LogisticRegression()
    meta_learner.fit(X_val_meta, y_val)
    
    # Collect predictions for all sets using Meta-Learner
    cp_val = meta_learner.predict_proba(X_val_meta)[:, 1]
    cp_test = meta_learner.predict_proba(np.column_stack(test_probs_all))[:, 1]
    cp_ext = meta_learner.predict_proba(np.column_stack(ext_probs_all))[:, 1]
    
    # THRESHOLDS
    logger.info("Optimizing thresholds...")
    candidates = np.linspace(0.01, 0.99, 100)
    best_ba, std_t = 0, 0.5
    for t in candidates:
        ba = balanced_accuracy_score(y_val, (cp_val >= t).astype(int))
        if ba > best_ba: best_ba, std_t = ba, t
    
    safety_t = std_t
    best_spec = 0
    for t in candidates:
        pred = (cp_val >= t).astype(int)
        if recall_score(y_val, pred) >= 0.85:
            spec = recall_score(y_val, pred, pos_label=0)
            if spec > best_spec: best_spec, safety_t = spec, t

    # FINAL EVALUATION + BOOTSTRAP
    logger.info("\nStarting Final Evaluation with Bootstrapping...")
    results = {'std': {}, 'safety': {}}
    for t_name, t_val in [('std', std_t), ('safety', safety_t)]:
        logger.info(f"  Evaluating threshold policy: {t_name} (t={t_val:.2f})")
        results[t_name]['internal_test'] = get_metrics_with_ci(y_test, cp_test, t_val)
        results[t_name]['external_ham'] = get_metrics_with_ci(y_ham, cp_ext, t_val)
        
    logger.info("Generating diagnostic plots...")
    plot_diagnostics(y_test, cp_test, y_ham, cp_ext, suffix="final_ensemble")
    
    # SAVE
    logger.info(f"Saving final results to {RESULTS_FILE}")
    with open(RESULTS_FILE, 'w') as f: json.dump(results, f, indent=4)
    bundle = {
        'models': models_ensemble, 
        'pca': pca, 
        'scaler': scaler, 
        'meta_learner': meta_learner, 
        'thresholds': {'std': std_t, 'safety': safety_t}
    }
    with open(MODEL_FILE, 'wb') as f: pickle.dump(bundle, f)
    
    logger.info(f"\nSUCCESS. Final PhD Bundle Saved. Results in {RESULTS_FILE.name}")
    logger.info(f"Plots saved in {PLOTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()
    
    try:
        run_phd_pipeline(smoke_test=args.smoke_test)
    except Exception as e:
        logger.exception("Pipeline failed with an error:")
        sys.exit(1)
