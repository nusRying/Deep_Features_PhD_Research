import sys
import json
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Setup Standalone Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
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

def run_v2(smoke_test=False):
    logger.info("--- PHASE 2: SAFETY LAYER (V2) ---")
    if smoke_test:
        logger.info("SMOKE TEST MODE")
        iters = 2000
        n_pop = 500
    else:
        iters = 300000
        n_pop = 3000
    
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
    
    # Split for threshold tuning
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Train Single Expert
    model = ExSTraCS(learning_iterations=iters, N=n_pop)
    model.fit(X_train, y_train)
    
    # Optimize Threshold on Val (Target high sensitivity)
    y_prob = model.predict_proba(X_val)[:, 1]
    best_t = 0.5
    best_ba = 0
    for t in np.linspace(0.05, 0.95, 50):
        ba = balanced_accuracy_score(y_val, (y_prob >= t).astype(int))
        if ba > best_ba:
            best_ba = ba
            best_t = t
            
    logger.info(f"Optimized Threshold: {best_t}")
    
    # Eval on Test
    y_prob_h = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob_h >= best_t).astype(int)
    results = {
        'ba': balanced_accuracy_score(y_test, y_pred),
        'sens': recall_score(y_test, y_pred),
        'spec': recall_score(y_test, y_pred, pos_label=0),
        'threshold': float(best_t)
    }
    
    with open(RESULTS_DIR / "results.json", 'w') as f: json.dump(results, f, indent=4)
    logger.info(f"V2 Results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v2(smoke_test=args.smoke_test)
