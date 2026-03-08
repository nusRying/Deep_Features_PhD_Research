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

# Unified Ph.D. Utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import compute_phd_metrics

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

def run_v1(smoke_test=False):
    logger.info("--- PHASE 1: BASELINE (V1) ---")
    if smoke_test:
        logger.info("SMOKE TEST MODE")
        iters = 2000
        n_pop = 500
    else:
        iters = 300000
        n_pop = 3000
    
    # Load Data (Rooted in shared data folder)
    df_feats = pd.read_csv(DATA_DIR / "features" / "hybrid_features_isic.csv")
    df_meta = pd.read_csv(DATA_DIR / "metadata" / "metadata_isic_clean.csv")
    df = pd.merge(df_feats, df_meta, on='image_id')
    
    X = df[[f"deep_{i}" for i in range(2048)]].values
    y = df['label'].values
    
    # Simple PCA to 256
    pca = PCA(n_components=256, random_state=42)
    X = pca.fit_transform(X)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train Single Expert
    model = ExSTraCS(learning_iterations=iters, N=n_pop)
    model.fit(X_train, y_train)
    
    # Eval
    y_pred = model.predict(X_test)
    results = compute_phd_metrics(y_test, y_pred)
    
    with open(RESULTS_DIR / "results.json", 'w') as f: json.dump(results, f, indent=4)
    logger.info(f"V1 Results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v1(smoke_test=args.smoke_test)
