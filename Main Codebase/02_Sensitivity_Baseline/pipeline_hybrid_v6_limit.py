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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, accuracy_score, 
    recall_score, precision_score, roc_curve, auc, precision_recall_curve,
    matthews_corrcoef, brier_score_loss
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.utils import resample

# Project Paths
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
RESULTS_FILE = RESULTS_DIR / "results_v6_limit.json"
MODEL_FILE = MODELS_DIR / "hybrid_ensemble_v6_limit.pkl"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints_v6"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Setup
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "v6_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 789, 2024, 99]
LEARNING_ITERATIONS = 500000 
N_POP = 5000
BOOTSTRAP_SAMPLES = 1000
ATTR_SUMS_FILE = ROOT_DIR / "attr_tracking_sums.npy"

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals[non_empty_bins] / len(y_prob)
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return float(ece)

def get_metrics_comprehensive(y_true, y_prob, threshold=0.5):
    """Calculates standard and PhD-grade metrics (MCC, Brier, ECE)."""
    def calc_metrics(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        return {
            "ba": balanced_accuracy_score(y_t, y_pred),
            "sens": recall_score(y_t, y_pred),
            "spec": recall_score(y_t, y_pred, pos_label=0),
            "mcc": matthews_corrcoef(y_t, y_pred),
            "brier": brier_score_loss(y_t, y_p),
            "ece": calculate_ece(y_t, y_p)
        }

    base_metrics = calc_metrics(y_true, y_prob)
    
    # Bootstrapping for CI
    boot_stats = {k: [] for k in base_metrics.keys()}
    for i in range(BOOTSTRAP_SAMPLES):
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
    return final_results

def run_v6_pipeline(smoke_test=False):
    global BOOTSTRAP_SAMPLES, LEARNING_ITERATIONS, N_POP
    if smoke_test:
        logger.info("SMOKE TEST MODE ENABLED")
        BOOTSTRAP_SAMPLES = 5
        LEARNING_ITERATIONS = 2000
        N_POP = 500
    
    logger.info("--- PHASE 4: THEORETICAL LIMIT (STACKING V3 + SMOTE-ENN) ---")
    
    # Data Loading
    df_feats = pd.read_csv(HYBRID_ISIC_FILE)
    df_meta = pd.read_csv(ISIC_METADATA_FILE)
    df = pd.merge(df_feats, df_meta, on='image_id', how='inner')
    
    deep_cols = [f"deep_{i}" for i in range(2048)]
    hand_meta_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in deep_cols and c != 'label']
    
    X_deep = df[deep_cols].values
    y = df['label'].values
    
    # PCA
    logger.info("Fitting PCA...")
    pca = PCA(n_components=256, random_state=42)
    X_deep_pca = pca.fit_transform(X_deep)
    X_raw = np.hstack([X_deep_pca, df[hand_meta_cols].values])
    
    # LCS-GUIDED PRUNING
    if ATTR_SUMS_FILE.exists():
        logger.info("Applying LCS-Guided Feature Pruning...")
        attr_sums = np.load(ATTR_SUMS_FILE)
        # Keep top 50% of features
        threshold_val = np.median(attr_sums)
        keep_indices = np.where(attr_sums > threshold_val)[0]
        X_pruned = X_raw[:, keep_indices]
        logger.info(f"  Pruned features from {X_raw.shape[1]} down to {X_pruned.shape[1]}")
    else:
        logger.warning("Attribute sums file not found. Skipping pruning.")
        X_pruned = X_raw
        keep_indices = np.arange(X_raw.shape[1])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_pruned)
    
    # SPLITS
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # EXTERNAL DATA
    df_ham = pd.merge(pd.read_csv(HYBRID_HAM_FILE), pd.read_csv(HAM_METADATA_FILE), on='image_id', how='inner')
    for col in hand_meta_cols:
        if col not in df_ham.columns: df_ham[col] = 0
    X_ham_raw = np.hstack([pca.transform(df_ham[deep_cols].values), df_ham[hand_meta_cols].values])
    X_ham = scaler.transform(X_ham_raw[:, keep_indices])
    y_ham = df_ham['label'].values
    
    # TIER 1: ENSEMBLE + OOF
    logger.info("Training Tier 1 Experts (5-Fold OOF)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta_features = np.zeros(len(y_train))
    models_tier1 = []
    
    # For actual production, we train 5 experts on full training set
    # For OOF, we simulate this by training on 4/5 of train set
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        ckpt_path = CHECKPOINT_DIR / f"expert_fold_{fold}.pkl"
        if ckpt_path.exists() and not smoke_test:
            logger.info(f"  Loading Expert for Fold {fold}...")
            with open(ckpt_path, 'rb') as f: model = pickle.load(f)
        else:
            logger.info(f"  Training Expert for Fold {fold} (SMOTE-ENN)...")
            Xt_f, yt_f = X_train[train_idx], y_train[train_idx]
            smote_enn = SMOTEENN(random_state=42)
            X_res, y_res = smote_enn.fit_resample(Xt_f, yt_f)
            
            model = ExSTraCS(learning_iterations=LEARNING_ITERATIONS, N=N_POP, nu=5)
            model.fit(X_res, y_res)
            with open(ckpt_path, 'wb') as f: pickle.dump(model, f)
            
        # Generate OOF predictions
        oof_meta_features[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
        models_tier1.append(model)
        if smoke_test and fold >= 1: break # Only 2 folds for smoke test

    # TIER 2 & 3: META-LEARNERS
    logger.info("Training Meta-Learners (Tier 2 & 3)...")
    # Meta Features for Val/Test/Ext
    def get_tier1_preds(X_data):
        preds = []
        for m in models_tier1:
            preds.append(m.predict_proba(X_data)[:, 1])
        return np.mean(preds, axis=0) # Simple mean is the base meta-feature

    X_train_meta = oof_meta_features.reshape(-1, 1)
    
    # Model A: Cost-Sensitive LR
    meta_a = LogisticRegression(class_weight={0: 1, 1: 2.0})
    meta_a.fit(X_train_meta, y_train)
    
    # Model B: Non-Linear RF
    meta_b = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    meta_b.fit(X_train_meta, y_train)
    
    # Tier 3 Aggregator: Isotonic Calibration on Weighted Ensemble
    p_val_a = meta_a.predict_proba(get_tier1_preds(X_val).reshape(-1, 1))[:, 1]
    p_val_b = meta_b.predict_proba(get_tier1_preds(X_val).reshape(-1, 1))[:, 1]
    p_val_agg = (p_val_a * 0.7 + p_val_b * 0.3)
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_val_agg, y_val)
    
    # FINAL INFERENCE
    def final_predict(X_data):
        p1 = get_tier1_preds(X_data).reshape(-1, 1)
        p2a = meta_a.predict_proba(p1)[:, 1]
        p2b = meta_b.predict_proba(p1)[:, 1]
        p_agg = (p2a * 0.7 + p2b * 0.3)
        return calibrator.transform(p_agg)

    cp_test = final_predict(X_test)
    cp_ext = final_predict(X_ham)
    
    # RESULTS
    logger.info("Final Evaluation...")
    best_ba, best_t = 0, 0.5
    p_val_final = calibrator.transform(p_val_agg)
    for t in np.linspace(0.01, 0.99, 100):
        ba = balanced_accuracy_score(y_val, (p_val_final >= t).astype(int))
        if ba > best_ba: best_ba, best_t = ba, t
        
    results = {
        'v6_limit': {
            'internal': get_metrics_comprehensive(y_test, cp_test, best_t),
            'external': get_metrics_comprehensive(y_ham, cp_ext, best_t)
        }
    }
    
    with open(RESULTS_FILE, 'w') as f: json.dump(results, f, indent=4)
    logger.info(f"RESULTS SAVED: {RESULTS_FILE}")
    
    # SAVE BUNDLE
    bundle = {
        'tier1': models_tier1, 'meta_a': meta_a, 'meta_b': meta_b, 
        'calibrator': calibrator, 'pca': pca, 'scaler': scaler, 
        'keep_indices': keep_indices, 'best_t': best_t
    }
    with open(MODEL_FILE, 'wb') as f: pickle.dump(bundle, f)
    logger.info(f"BUNDLE SAVED: {MODEL_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v6_pipeline(smoke_test=args.smoke_test)
