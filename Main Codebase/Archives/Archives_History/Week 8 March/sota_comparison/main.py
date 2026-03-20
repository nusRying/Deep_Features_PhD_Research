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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, accuracy_score, 
    recall_score, precision_score, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
# ROOT_DIR points to Deep_Features_Experiment (SOTA is in Week 8 March, so parent.parent)
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
RESULTS_FILE = RESULTS_DIR / "results_sota.json"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "sota_comparison.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_sota_comparison(smoke_test=False):
    logger.info("--- WEEK 8 MARCH: SOTA MODELS COMPARISON ---")
    
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
    X_ham = df_ham_feats[deep_cols].values

    if smoke_test:
        logger.info("SMOKE TEST MODE: Reducing dataset size")
        X, y = X[:500], y[:500]
        X_ham, y_ham = X_ham[:200], y_ham[:200]

    # Preprocessing
    logger.info(f"Scaling {X.shape[1]} features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_ham_scaled = scaler.transform(X_ham)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Models definition
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=10 if smoke_test else 100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=10 if smoke_test else 100, random_state=42, eval_metric='logloss', n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=10 if smoke_test else 100, random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=10 if smoke_test else 100, random_state=42, verbose=0, thread_count=-1),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }

    all_results = {}
    roc_data = {'Internal': {}, 'External': {}}
    pr_data = {'Internal': {}, 'External': {}}

    for name, model in models.items():
        logger.info(f"--- Training {name} ---")
        start_time = time.time()
        model.fit(X_train, y_train)
        logger.info(f"{name} Training Complete. Time: {time.time() - start_time:.2f}s")
        
        # Internal Eval
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        res_int = {
            "ba": float(balanced_accuracy_score(y_test, y_test_pred)),
            "sens": float(recall_score(y_test, y_test_pred)),
            "spec": float(recall_score(y_test, y_test_pred, pos_label=0)),
            "acc": float(accuracy_score(y_test, y_test_pred))
        }
        
        cm_int = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix - Internal')
        plt.savefig(PLOTS_DIR / f"cm_{name.lower()}_internal.png")
        plt.close()
        
        # External Eval
        y_ham_pred = model.predict(X_ham_scaled)
        y_ham_prob = model.predict_proba(X_ham_scaled)[:, 1]
        
        res_ext = {
            "ba": float(balanced_accuracy_score(y_ham, y_ham_pred)),
            "sens": float(recall_score(y_ham, y_ham_pred)),
            "spec": float(recall_score(y_ham, y_ham_pred, pos_label=0)),
            "acc": float(accuracy_score(y_ham, y_ham_pred))
        }
        
        cm_ext = confusion_matrix(y_ham, y_ham_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix - External')
        plt.savefig(PLOTS_DIR / f"cm_{name.lower()}_external.png")
        plt.close()
        
        all_results[name] = {"internal": res_int, "external": res_ext}
        
        # Collect data for overlaid plots
        fpr_int, tpr_int, _ = roc_curve(y_test, y_test_prob)
        roc_data['Internal'][name] = (fpr_int, tpr_int, auc(fpr_int, tpr_int))
        
        precision_int, recall_int, _ = precision_recall_curve(y_test, y_test_prob)
        pr_data['Internal'][name] = (recall_int, precision_int)
        
        fpr_ext, tpr_ext, _ = roc_curve(y_ham, y_ham_prob)
        roc_data['External'][name] = (fpr_ext, tpr_ext, auc(fpr_ext, tpr_ext))
        
        precision_ext, recall_ext, _ = precision_recall_curve(y_ham, y_ham_prob)
        pr_data['External'][name] = (recall_ext, precision_ext)
        

    # Plot Overlaid ROC and PR curves
    for dataset in ['Internal', 'External']:
        # ROC
        plt.figure(figsize=(10, 8))
        for name, (fpr, tpr, roc_auc) in roc_data[dataset].items():
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'SOTA ROC Curves - {dataset}')
        plt.legend(loc="lower right")
        plt.savefig(PLOTS_DIR / f"roc_overlaid_{dataset.lower()}.png")
        plt.close()
        
        # PR
        plt.figure(figsize=(10, 8))
        for name, (rec, prec) in pr_data[dataset].items():
            plt.plot(rec, prec, lw=2, label=name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'SOTA PR Curves - {dataset}')
        plt.legend(loc="lower left")
        plt.savefig(PLOTS_DIR / f"pr_overlaid_{dataset.lower()}.png")
        plt.close()

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"SOTA Comparison complete. Results saved to {RESULTS_FILE}")
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    run_sota_comparison(smoke_test=args.smoke_test)
