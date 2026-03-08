import sys
import json
import numpy as np
import pandas as pd
import pickle
import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score, matthews_corrcoef, brier_score_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import IsotonicRegression, calibration_curve
from imblearn.combine import SMOTEENN

# Setup Standalone Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
LOG_DIR = SCRIPT_DIR / "logs"
EXTERNAL_DIR = SCRIPT_DIR / "external"
sys.path.insert(0, str(EXTERNAL_DIR / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

# Data Files
HYBRID_ISIC_FILE = DATA_DIR / "features" / "hybrid_features_isic.csv"
ISIC_METADATA_FILE = DATA_DIR / "metadata" / "metadata_isic_clean.csv"
ATTR_SUMS_FILE = SCRIPT_DIR / "attr_tracking_sums.npy"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_DIR / "v6.log"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 789, 2024, 99]

def run_v6(smoke_test=False):
    logger.info("--- PHASE 6: GLOBAL PEAK (V6) ---")
    if smoke_test: logger.info("SMOKE TEST MODE")
    
    # Load Data
    df_feats = pd.read_csv(HYBRID_ISIC_FILE)
    df_meta = pd.read_csv(ISIC_METADATA_FILE)
    df = pd.merge(df_feats, df_meta, on='image_id')
    
    deep_cols = [f"deep_{i}" for i in range(2048)]
    hand_meta_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in deep_cols and c != 'label']
    X_deep = df[deep_cols].values
    y = df['label'].values
    
    # PCA
    pca = PCA(n_components=256, random_state=42)
    X_raw = np.hstack([pca.fit_transform(X_deep), df[hand_meta_cols].values])
    
    # LCS-GUIDED PRUNING (50% noise removal)
    if ATTR_SUMS_FILE.exists():
        attr_sums = np.load(ATTR_SUMS_FILE)
        X_pruned = X_raw[:, np.where(attr_sums > np.median(attr_sums))[0]]
        logger.info(f"Pruned features: {X_raw.shape[1]} -> {X_pruned.shape[1]}")
    else:
        X_pruned = X_raw

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_pruned)
    
    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Tier 1 (OOF Stacking)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.zeros(len(y_train))
    experts = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"Fold {fold+1}/5...")
        Xt_f, yt_f = X_train[t_idx], y_train[t_idx]
        X_res, y_res = SMOTEENN(random_state=42).fit_resample(Xt_f, yt_f)
        
        model = ExSTraCS(learning_iterations=2000 if smoke_test else 500000, N=5000)
        model.fit(X_res, y_res)
        oof_meta[v_idx] = model.predict_proba(X_train[v_idx])[:, 1]
        experts.append(model)
        if smoke_test and fold >= 0: break 

    # Tier 2 & 3: Meta-Committee
    X_train_meta = oof_meta.reshape(-1, 1)
    meta_a = LogisticRegression(class_weight={0: 1, 1: 2.0}).fit(X_train_meta, y_train)
    meta_b = RandomForestClassifier(n_estimators=100, max_depth=3).fit(X_train_meta, y_train)
    
    # Aggregator Calibration
    def get_tier1_preds(X_data): return np.mean([m.predict_proba(X_data)[:, 1] for m in experts], axis=0)
    p_val_a = meta_a.predict_proba(get_tier1_preds(X_val).reshape(-1, 1))[:, 1]
    p_val_b = meta_b.predict_proba(get_tier1_preds(X_val).reshape(-1, 1))[:, 1]
    p_val_agg = (p_val_a * 0.7 + p_val_b * 0.3)
    
    calibrator = IsotonicRegression(out_of_bounds='clip').fit(p_val_agg, y_val)
    
    # Final Eval
    p_test_tier1 = get_tier1_preds(X_test).reshape(-1, 1)
    p_test_a = meta_a.predict_proba(p_test_tier1)[:, 1]
    p_test_b = meta_b.predict_proba(p_test_tier1)[:, 1]
    cp_test = calibrator.transform(p_test_a * 0.7 + p_test_b * 0.3)
    
    results = {
        'ba': balanced_accuracy_score(y_test, (cp_test >= 0.5).astype(int)),
        'mcc': matthews_corrcoef(y_test, (cp_test >= 0.5).astype(int)),
        'brier': brier_score_loss(y_test, cp_test)
    }
    
    with open(RESULTS_DIR / "results.json", 'w') as f: json.dump(results, f, indent=4)
    logger.info(f"V6 Results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v6(smoke_test=args.smoke_test)
