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

def run_v8_latent(smoke_test=False):
    logger.info("--- PHASE 8d: LKH-LCS (LATENT KNOWLEDGE HARVESTING) ---")
    if smoke_test:
        logger.info("SMOKE TEST MODE - Validating Latent Archive Mechanics")
        iters = 50000 # Need enough iterations for the GA to start deleting rules and filling the archive
        n_pop = 100 # Artificially small population cap to FORCE catastrophic forgetting and trigger the archive
    else:
        iters = 500000 
        n_pop = 2000 # Standard population size
        
    logger.info(f"Loading data from {DATA_DIR}...")
    df_feats = pd.read_csv(DATA_DIR / "features" / "hybrid_features_isic.csv")
    df_meta = pd.read_csv(DATA_DIR / "metadata" / "metadata_isic_clean.csv")
    df = pd.merge(df_feats, df_meta, on='image_id')
    
    # Feature Extraction (Hybrid)
    X = df[[f"deep_{i}" for i in range(2048)]].values
    y = df['label'].values
    
    # Scale and Reduce
    pca = PCA(n_components=64, random_state=42)
    X = pca.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize Phase 8d Architecture
    logger.info(f"Training LKH-LCS ({iters} Iterations, tight N={n_pop} to force archiving)...")
    # Using tracking so we can observe rule dynamics
    model = ExSTraCS(learning_iterations=iters, N=n_pop, random_state=42, track_accuracy_while_fit=True)

    try:
        model.fit(X_train, y_train)
        
        archive_size = len(model.population.latent_archive)
        logger.info(f"Latent Archive populated completely: {archive_size} highly-accurate rules salvaged from deletion!")
        
        logger.info("Extracting Classifications against active AND latent sets...")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        rescued = getattr(model, 'rescued_predictions', 0)
        logger.info(f"Predictions Rescued directly by Latent Archive: {rescued} / {len(X_test)} samples")
        
        # Unified Auditing
        results = compute_phd_metrics(y_test, y_pred, y_prob)
        results["archive_size"] = archive_size
        results["rescued_predictions"] = rescued
        
        with open(RESULTS_DIR / "results.json", 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Phase 8d LKH-LCS Balanced Accuracy: {results['ba']:.4f}")
        logger.info(f"Phase 8d LKH-LCS Minority Sensitivity: {results['sens']:.4f}")

    except Exception as e:
        logger.error(f"Critical architecture failure during Archive integration: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v8_latent(smoke_test=args.smoke_test)
