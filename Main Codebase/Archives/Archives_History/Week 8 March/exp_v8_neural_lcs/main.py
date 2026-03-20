import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import argparse

# Unified Ph.D. Utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import compute_phd_metrics

# Setup Standalone Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
EXTERNAL_DIR = SCRIPT_DIR / "external"
sys.path.insert(0, str(EXTERNAL_DIR / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

# Ensure directories
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_v8_neural(smoke_test=False):
    logger.info("--- PHASE 8c: MODULAR NEURAL-LCS (MN-LCS) ---")
    if smoke_test:
        logger.info("SMOKE TEST MODE - Validating Micro-MLP Injection")
        iters = 5000
        n_pop = 500
    else:
        iters = 250000 
        n_pop = 2500 # Slightly smaller pop needed due to denser neural knowledge
        
    logger.info(f"Loading data from {DATA_DIR}...")
    df_feats = pd.read_csv(DATA_DIR / "features" / "hybrid_features_isic.csv")
    df_meta = pd.read_csv(DATA_DIR / "metadata" / "metadata_isic_clean.csv")
    df = pd.merge(df_feats, df_meta, on='image_id')
    
    # Feature Extraction (Hybrid)
    X = df[[f"deep_{i}" for i in range(2048)]].values
    y = df['label'].values
    
    # Scale and Reduce (Crucial for Neural Sub-Agents)
    pca = PCA(n_components=256, random_state=42)
    X = pca.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize Phase 8c Architecture
    logger.info(f"Training MN-LCS with NCL ({iters} Iterations, N={n_pop})...")
    model = ExSTraCS(learning_iterations=iters, N=n_pop, random_state=42)

    try:
        model.fit(X_train, y_train)
        
        logger.info("Extracting Classifications via Micro-Neural Ensembles...")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Unified Auditing
        results = compute_phd_metrics(y_test, y_pred, y_prob)
        
        with open(RESULTS_DIR / "results.json", 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Phase 8c MN-LCS Balanced Accuracy: {results['ba']:.4f}")
        logger.info(f"Phase 8c LKH-LCS Specificity: {results['spec']:.4f}")

    except Exception as e:
        logger.error(f"Critical architecture failure during Neural integration: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v8_neural(smoke_test=args.smoke_test)
