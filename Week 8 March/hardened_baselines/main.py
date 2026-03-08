import sys
import json
import numpy as np
import pandas as pd
import time
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, accuracy_score, 
    recall_score, precision_score, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent 
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "Plots"
LOG_DIR = SCRIPT_DIR / "logs"

# Data Files
HYBRID_ISIC_FILE = DATA_DIR / "features" / "hybrid_features_isic.csv"
HYBRID_HAM_FILE = DATA_DIR / "features" / "hybrid_features_ham.csv"
ISIC_METADATA_FILE = DATA_DIR / "metadata" / "metadata_isic_clean.csv"
HAM_METADATA_FILE = DATA_DIR / "metadata" / "metadata_ham_clean.csv"

# Output Files
RESULTS_FILE = RESULTS_DIR / "results_hardened.json"
STATS_CSV = RESULTS_DIR / "stats_hardened.csv"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "hardened_baselines.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_hardened_baselines(smoke_test=False):
    logger.info("--- WEEK 8 MARCH: HARDENED BASELINES (PCA + K-FOLD + ENSEMBLE) ---")
    
    # 1. Data Loading
    logger.info("Loading Data...")
    df_feats = pd.read_csv(HYBRID_ISIC_FILE)
    df_meta = pd.read_csv(ISIC_METADATA_FILE)
    df = pd.merge(df_feats, df_meta, on='image_id', how='inner')
    
    deep_cols = [f"deep_{i}" for i in range(2048)]
    X = df[deep_cols].values
    y = df['label'].values
    
    df_ham_feats = pd.read_csv(HYBRID_HAM_FILE)
    y_ham = pd.read_csv(HAM_METADATA_FILE)['label'].values
    X_ham_raw = df_ham_feats[deep_cols].values

    if smoke_test:
        logger.info("SMOKE TEST MODE: Reducing dataset size")
        X, y = X[:1000], y[:1000]
        X_ham_raw, y_ham = X_ham_raw[:500], y_ham[:500]

    # 2. PCA Normalization (Matches v6 Pipeline)
    logger.info(f"Scaling and applying PCA (n=256) to {X.shape[1]} features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_ham_scaled = scaler.transform(X_ham_raw)
    
    pca = PCA(n_components=256, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_ham_pca = pca.transform(X_ham_scaled)
    logger.info(f"Variance explained by 256 components: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 3. Model Definitions
    base_models = [
        ("RF", RandomForestClassifier(n_estimators=50 if smoke_test else 100, random_state=42, n_jobs=-1)),
        ("XGB", xgb.XGBClassifier(n_estimators=50 if smoke_test else 100, random_state=42, eval_metric='logloss', n_jobs=-1)),
        ("LGBM", lgb.LGBMClassifier(n_estimators=50 if smoke_test else 100, random_state=42, n_jobs=-1, verbose=-1)),
        ("CB", CatBoostClassifier(iterations=50 if smoke_test else 100, random_state=42, verbose=0, thread_count=-1)),
        ("SVM", SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    ensemble = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
    all_models = base_models + [("SOTA_Ensemble", ensemble)]

    # 4. K-Fold Cross-Validation Routine
    skf = StratifiedKFold(n_splits=3 if smoke_test else 5, shuffle=True, random_state=42)
    final_stats = []

    for name, model in all_models:
        logger.info(f"Evaluating Model: {name} (K-Fold + External)")
        fold_results = {"ba": [], "sens": [], "spec": []}
        ext_results = {"ba": [], "sens": [], "spec": []}
        
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca, y)):
            X_train, X_val = X_pca[train_idx], X_pca[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            
            # Internal (Fold) Eval
            y_pred = model.predict(X_val)
            fold_results["ba"].append(balanced_accuracy_score(y_val, y_pred))
            fold_results["sens"].append(recall_score(y_val, y_pred))
            fold_results["spec"].append(recall_score(y_val, y_pred, pos_label=0))
            
            # External (HAM) Eval (Incremental validation)
            y_ext_pred = model.predict(X_ham_pca)
            ext_results["ba"].append(balanced_accuracy_score(y_ham, y_ext_pred))
            ext_results["sens"].append(recall_score(y_ham, y_ext_pred))
            ext_results["spec"].append(recall_score(y_ham, y_ext_pred, pos_label=0))
            
        duration = time.time() - start_time
        logger.info(f"{name} K-Fold Complete in {duration:.2f}s")
        
        # Calculate Stats (Mean ± Std)
        stats = {
            "Model": name,
            "Internal_BA": f"{np.mean(fold_results['ba']):.4f} ± {np.std(fold_results['ba']):.4f}",
            "Internal_Sens": f"{np.mean(fold_results['sens']):.4f} ± {np.std(fold_results['sens']):.4f}",
            "Internal_Spec": f"{np.mean(fold_results['spec']):.4f} ± {np.std(fold_results['spec']):.4f}",
            "External_BA": f"{np.mean(ext_results['ba']):.4f} ± {np.std(ext_results['ba']):.4f}",
            "External_Sens": f"{np.mean(ext_results['sens']):.4f} ± {np.std(ext_results['sens']):.4f}",
            "External_Spec": f"{np.mean(ext_results['spec']):.4f} ± {np.std(ext_results['spec']):.4f}",
        }
        final_stats.append(stats)

    # 5. Save Results
    df_stats = pd.DataFrame(final_stats)
    df_stats.to_csv(STATS_CSV, index=False)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_stats, f, indent=4)
        
    logger.info(f"Statistical validation complete. Stats saved to {STATS_CSV}")
    print("\n--- HARDENED BASELINES STATISTICAL SUMMARY ---")
    print(df_stats[["Model", "Internal_BA", "External_BA"]].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    run_hardened_baselines(smoke_test=args.smoke_test)
