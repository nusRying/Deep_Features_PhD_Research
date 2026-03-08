import sys
import json
import numpy as np
import pandas as pd
import pickle
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
from sklearn.neural_network import MLPClassifier

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent # Deep_Features_Experiment
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
LOG_DIR = SCRIPT_DIR / "logs"

# Data Files
HYBRID_ISIC_FILE = DATA_DIR / "features" / "hybrid_features_isic.csv"
HYBRID_HAM_FILE = DATA_DIR / "features" / "hybrid_features_ham.csv"
ISIC_METADATA_FILE = DATA_DIR / "metadata" / "metadata_isic_clean.csv"
HAM_METADATA_FILE = DATA_DIR / "metadata" / "metadata_ham_clean.csv"

# Output Files
RESULTS_FILE = RESULTS_DIR / "results.json"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "dl_baseline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def save_plots(y_true, y_prob, name, folder):
    """Generates and saves ROC and PR curves."""
    plt.figure(figsize=(12, 5))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    
    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {name}')
    
    plt.tight_layout()
    plt.savefig(folder / f"plots_{name.lower()}.png")
    plt.close()

def save_confusion_matrix(y_true, y_pred, name, folder):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(folder / f"cm_{name.lower()}.png")
    plt.close()

def run_dl_baseline(smoke_test=False):
    logger.info("--- WEEK MARCH 5: RAW DEEP LEARNING BASELINE (DETAILED) ---")
    
    # 1. Data Loading
    logger.info("Loading ISIC data...")
    df_feats = pd.read_csv(HYBRID_ISIC_FILE)
    df_meta = pd.read_csv(ISIC_METADATA_FILE)
    df = pd.merge(df_feats, df_meta, on='image_id', how='inner')
    
    deep_cols = [f"deep_{i}" for i in range(2048)]
    X = df[deep_cols].values
    y = df['label'].values
    
    if smoke_test:
        logger.info("SMOKE TEST MODE: Reducing dataset size")
        X, y = X[:500], y[:500]

    # 2. Preprocessing
    logger.info(f"Scaling {X.shape[1]} features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Model
    logger.info("Initializing MLP Model...")
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 128),
        max_iter=5 if smoke_test else 200,
        random_state=42,
        early_stopping=True,
        verbose=True
    )
    
    logger.info("Training Model...")
    clf.fit(X_train, y_train)
    
    # 4. Detailed Evaluation
    logger.info("Running Detailed Evaluation...")
    
    # Plots Folder
    PLOTS_DIR = RESULTS_DIR / "Plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Internal Performance
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    
    results = {
        "internal": {
            "ba": float(balanced_accuracy_score(y_test, y_test_pred)),
            "sens": float(recall_score(y_test, y_test_pred)),
            "spec": float(recall_score(y_test, y_test_pred, pos_label=0)),
            "acc": float(accuracy_score(y_test, y_test_pred))
        }
    }
    
    save_plots(y_test, y_test_prob, "Internal", PLOTS_DIR)
    save_confusion_matrix(y_test, y_test_pred, "Internal", PLOTS_DIR)
    
    # External Performance
    try:
        logger.info("Loading HAM10000 for External Verification...")
        df_ham_feats = pd.read_csv(HYBRID_HAM_FILE)
        y_ham = pd.read_csv(HAM_METADATA_FILE)['label'].values
        X_ham = scaler.transform(df_ham_feats[deep_cols].values)
        
        y_ham_pred = clf.predict(X_ham)
        y_ham_prob = clf.predict_proba(X_ham)[:, 1]
        
        results["external"] = {
            "ba": float(balanced_accuracy_score(y_ham, y_ham_pred)),
            "sens": float(recall_score(y_ham, y_ham_pred)),
            "spec": float(recall_score(y_ham, y_ham_pred, pos_label=0)),
            "acc": float(accuracy_score(y_ham, y_ham_pred))
        }
        
        save_plots(y_ham, y_ham_prob, "External", PLOTS_DIR)
        save_confusion_matrix(y_ham, y_ham_pred, "External", PLOTS_DIR)
        
    except Exception as e:
        logger.error(f"External validation failed: {e}")
    
    # 5. Systematic Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"ALL RESULTS & GRAPHS SAVED TO: {RESULTS_DIR}")
    print("\n--- BASELINE DL SUMMARY ---")
    print(f"Internal BA: {results['internal']['ba']:.4f}")
    if "external" in results:
        print(f"External BA: {results['external']['ba']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_dl_baseline(smoke_test=args.smoke_test)
