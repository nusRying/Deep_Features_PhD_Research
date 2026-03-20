import sys
import json
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN

# Setup Standalone Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
EXTERNAL_DIR = SCRIPT_DIR / "external"
sys.path.insert(0, str(EXTERNAL_DIR / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

# Ensure directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import argparse

def run_v5(smoke_test=False):
    logger.info("--- PHASE 5: UNIVERSAL BALANCE (V5) ---")
    if smoke_test:
        logger.info("SMOKE TEST MODE")
        iters = 2000
        n_pop = 500
        n_splits = 2
    else:
        iters = 300000
        n_pop = 3000
        n_splits = 3
    
    # Load Data
    df_feats = pd.read_csv(DATA_DIR / "features" / "hybrid_features_isic.csv")
    df_meta = pd.read_csv(DATA_DIR / "metadata" / "metadata_isic_clean.csv")
    df = pd.merge(df_feats, df_meta, on='image_id')
    
    X = df[[f"deep_{i}" for i in range(2048)]].values
    y = df['label'].values
    
    # PCA + Scaler
    pca = PCA(n_components=256, random_state=42)
    X = pca.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # OOF Stacking with SMOTE-ENN
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_meta = np.zeros(len(y_train))
    experts = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"Fold {fold+1}/{n_splits}: SMOTE-ENN Balancing...")
        Xt_f, yt_f = X_train[t_idx], y_train[t_idx]
        from imblearn.combine import SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        X_res, y_res = smote_enn.fit_resample(Xt_f, yt_f)
        
        model = ExSTraCS(learning_iterations=iters, N=n_pop)
        model.fit(X_res, y_res)
        oof_meta[v_idx] = model.predict_proba(X_train[v_idx])[:, 1]
        experts.append(model)
        
    # Meta-Learner
    meta_learner = LogisticRegression()
    meta_learner.fit(oof_meta.reshape(-1, 1), y_train)
    
    # Eval
    test_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in experts], axis=0)
    y_pred = meta_learner.predict(test_probs.reshape(-1, 1))
    
    results = {
        'ba': balanced_accuracy_score(y_test, y_pred),
        'sens': recall_score(y_test, y_pred),
        'spec': recall_score(y_test, y_pred, pos_label=0)
    }
    
    with open(RESULTS_DIR / "results.json", 'w') as f: json.dump(results, f, indent=4)
    logger.info(f"V5 Results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v5(smoke_test=args.smoke_test)
