import sys
import os
from pathlib import Path
# Ensure project_paths.py is found at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


"""
Diagnostic Experiment: Deep Feature Sanity Check
------------------------------------------------
Goal: Prove that the selected Deep Features contain meaningful structure relative to random features.

Experiments:
1. Train on Top 200 Selected Features (Baseline)
2. Train on Random 200 Features (Control)
3. Train on Top 200 with 10% Swapped (Perturbation)

Output: diagnostic_report.json
"""

import sys
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from project_paths import (
    EXSTRACS_LIB_DIR,
    DEEP_FEATURES_FILE,
    FEATURE_SELECTION_METADATA_FILE,
    CUSTOM_LABELS_FILE,
    DIAGNOSTIC_REPORT_FILE,
)

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

# Config
FEATURE_FILE = DEEP_FEATURES_FILE
METADATA_FILE = FEATURE_SELECTION_METADATA_FILE
LABEL_SOURCE_PATH = CUSTOM_LABELS_FILE
RANDOM_SEED = 42
ITERATIONS = 1000 # Fast verification path-check

def train_and_eval(X, y, name):
    print(f"\n--- Experiment: {name} ---")
    print(f"  Feature Shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    # Normalize (Critical)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train
    model = ExSTraCS(learning_iterations=ITERATIONS, N=2000, nu=10, track_accuracy_while_fit=False)
    
    start = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  Error: {e}")
        return None

    duration = time.time() - start
    
    # Eval
    y_pred = model.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # ExSTraCS rule count access fix
    try:
        rules = model.get_final_rule_population_size()
    except AttributeError:
        rules = len(model.rule_population) if hasattr(model, 'rule_population') else -1
    
    print(f"  Result: BA={ba:.4f}, Acc={acc:.4f}, Rules={rules}, Time={duration:.1f}s")
    
    return {
        "balanced_accuracy": ba,
        "accuracy": acc,
        "rule_count": rules,
        "duration": duration
    }

def run_diagnostics():
    # 1. Load Data
    if not METADATA_FILE.exists():
        print(f"Error: {METADATA_FILE} missing.")
        return

    with open(METADATA_FILE) as f:
        meta = json.load(f)
        top_indices = meta['top_indices']
        
    print("Loading Features...")
    df_feats = pd.read_csv(FEATURE_FILE)
    df_labels = pd.read_csv(LABEL_SOURCE_PATH)
    df_merged = pd.merge(df_feats, df_labels[['image', 'label']], left_on='image_id', right_on='image', how='inner')
    
    drop_cols = ['image_id', 'image', 'label']
    valid_cols = [c for c in df_merged.columns if c not in drop_cols]
    
    X_full = df_merged[valid_cols].values
    y = df_merged['label'].values
    
    results = {}
    
    # Exp 1: Selected Features
    print("Running Baseline (Selected Features)...")
    X_selected = X_full[:, top_indices]
    results['Selected_Top200'] = train_and_eval(X_selected, y, "Selected Top 200")
    
    # Exp 2: Random Features
    print("Running Control (Random Features)...")
    np.random.seed(RANDOM_SEED)
    random_indices = np.random.choice(len(valid_cols), size=len(top_indices), replace=False)
    X_random = X_full[:, random_indices]
    results['Random_200'] = train_and_eval(X_random, y, "Random 200")
    
    # Exp 3: Perturbed (Selected but shuffled) - Optional check
    # Let's shuffle 10% of columns in X_selected to noise
    print("Running Perturbation (10% Noise)...")
    X_perturbed = X_selected.copy()
    n_pert = int(0.1 * X_perturbed.shape[1])
    cols_to_shuffle = np.random.choice(X_perturbed.shape[1], n_pert, replace=False)
    for c in cols_to_shuffle:
        np.random.shuffle(X_perturbed[:, c])
        
    results['Perturbed_10pct'] = train_and_eval(X_perturbed, y, "Perturbed 10%")
    
    # Save Report
    with open(DIAGNOSTIC_REPORT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDiagnostics Complete. Saved to diagnostic_report.json")

if __name__ == "__main__":
    run_diagnostics()
