import sys
import json
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from project_paths import (
    EXSTRACS_LIB_DIR,
    HYBRID_ISIC_FILE,
    ISIC_METADATA_FILE,
    HYBRID_HAM_FILE,
    HAM_METADATA_FILE,
    RESULTS_DIR,
    MODELS_DIR,
)

# Setup Paths
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

# Output Files
RESULTS_FILE = RESULTS_DIR / "results_hybrid_v3_ensemble.json"
MODEL_FILE = MODELS_DIR / "hybrid_ensemble_v3.pkl"

SEEDS = [42, 123, 789, 2024, 99]
LEARNING_ITERATIONS = 500000 # Keeping 500k for now to avoid 24h+ runs, but across 5 seeds it's 2.5M total.
N_POP = 3000

def evaluate_with_threshold(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ba": float(balanced_accuracy_score(y_true, y_pred)),
        "sens": float(recall_score(y_true, y_pred)),
        "spec": float(recall_score(y_true, y_pred, pos_label=0)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

def calibrate_probs(y_val, p_val, p_test, p_ext):
    print("Fitting Hybrid Calibrator (Platt)...")
    calibrator = LogisticRegression()
    calibrator.fit(p_val.reshape(-1, 1), y_val)
    
    cp_val = calibrator.predict_proba(p_val.reshape(-1, 1))[:, 1]
    cp_test = calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]
    cp_ext = calibrator.predict_proba(p_ext.reshape(-1, 1))[:, 1]
    
    return cp_val, cp_test, cp_ext, calibrator

def run_refined_pipeline():
    print("--- PHD REFINEMENT: ENSEMBLE STABILIZED PIPELINE (v3) ---")
    
    print("Loading Data...")
    df_feats = pd.read_csv(HYBRID_ISIC_FILE)
    df_meta = pd.read_csv(ISIC_METADATA_FILE)
    df = pd.merge(df_feats, df_meta, on='image_id', how='inner')
    
    # Preprocessing
    df = df.drop(columns=['lesion_id', 'dx', 'image', 'anatom_site_general', 'localization'], errors='ignore')
    deep_cols = [f"deep_{i}" for i in range(2048)]
    other_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    hand_meta_cols = [c for c in other_numeric if c not in deep_cols and c != 'label']
    
    X_deep = df[deep_cols].values
    X_hand_meta = df[hand_meta_cols].values
    y = df['label'].values
    
    print(f"Applying PCA (256 components)...")
    pca = PCA(n_components=256, random_state=42)
    X_deep_pca = pca.fit_transform(X_deep)
    X = np.hstack([X_deep_pca, X_hand_meta])
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # External Data Prep
    print("Preparing External Validation (HAM10000)...")
    df_ham_f = pd.read_csv(HYBRID_HAM_FILE)
    df_ham_m = pd.read_csv(HAM_METADATA_FILE)
    df_ham = pd.merge(df_ham_f, df_ham_m, on='image_id', how='inner')
    for col in hand_meta_cols:
        if col not in df_ham.columns: df_ham[col] = 0
    X_ham_deep = df_ham[deep_cols].values
    X_ham_pca = pca.transform(X_ham_deep)
    X_ham = np.hstack([X_ham_pca, df_ham[hand_meta_cols].values])
    X_ham = scaler.transform(X_ham)
    y_ham = df_ham['label'].values
    
    # ENSEMBLE TRAINING
    models_ensemble = []
    val_probs_list = []
    test_probs_list = []
    ext_probs_list = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\nTraining Expert {i+1}/5 (Seed: {seed})...")
        ros = RandomOverSampler(random_state=seed)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        
        model = ExSTraCS(learning_iterations=LEARNING_ITERATIONS, N=N_POP, nu=10)
        model.fit(X_res, y_res)
        models_ensemble.append(model)
        
        val_probs_list.append(model.predict_proba(X_val)[:, 1])
        test_probs_list.append(model.predict_proba(X_test)[:, 1])
        ext_probs_list.append(model.predict_proba(X_ham)[:, 1])
        
    # Aggregate Probabilities
    p_val_avg = np.mean(val_probs_list, axis=0)
    p_test_avg = np.mean(test_probs_list, axis=0)
    p_ext_avg = np.mean(ext_probs_list, axis=0)
    
    # CALIBRATION
    cp_val, cp_test, cp_ext, calibrator = calibrate_probs(y_val, p_val_avg, p_test_avg, p_ext_avg)
    
    # THRESHOLD TUNING (Robust Multi-Objective)
    print("\nTuning Thresholds (Safety Guarded)...")
    candidates = np.linspace(0.05, 0.95, 181)
    # 1. Standard (Max BA on Val)
    std_t = 0.5
    best_ba = 0
    for t in candidates:
        res = evaluate_with_threshold(y_val, cp_val, t)
        if res['ba'] > best_ba:
            best_ba = res['ba']
            std_t = t
            
    # 2. Safety (Target Sens >= 0.85 AND Max Spec)
    safety_t = std_t
    best_spec = 0
    for t in candidates:
        res = evaluate_with_threshold(y_val, cp_val, t)
        if res['sens'] >= 0.85:
            if res['spec'] > best_spec:
                best_spec = res['spec']
                safety_t = t
                
    print(f"Standard Threshold: {std_t:.2f}")
    print(f"Safety Threshold:   {safety_t:.2f} (Val Spec: {best_spec:.4f})")
    
    # RESULTS COMPILATION
    results = {'std': {}, 'safety': {}}
    for t_name, t_val in [('std', std_t), ('safety', safety_t)]:
        results[t_name]['val'] = evaluate_with_threshold(y_val, cp_val, t_val)
        results[t_name]['test'] = evaluate_with_threshold(y_test, cp_test, t_val)
        results[t_name]['external'] = evaluate_with_threshold(y_ham, cp_ext, t_val)
        
    # LOGGING
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"EXT BA (Safety): {results['safety']['external']['ba']:.4f}")
    print(f"EXT Sens (Safety): {results['safety']['external']['sens']:.4f}")
    print(f"EXT Spec (Safety): {results['safety']['external']['spec']:.4f}")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    # SAVE BUNDLE
    bundle = {
        'models': models_ensemble,
        'pca': pca,
        'scaler': scaler,
        'calibrator': calibrator,
        'feature_cols': hand_meta_cols,
        'thresholds': {'std': std_t, 'safety': safety_t}
    }
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(bundle, f)
        
    print(f"\nSUCCESS. Ensemble model saved to {MODEL_FILE.name}")

if __name__ == "__main__":
    run_refined_pipeline()
