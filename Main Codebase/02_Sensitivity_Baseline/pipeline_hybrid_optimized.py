
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
from project_paths import (
    EXSTRACS_LIB_DIR,
    HYBRID_ISIC_FILE,
    ISIC_METADATA_FILE,
    HYBRID_HAM_FILE,
    HAM_METADATA_FILE,
    HYBRID_RESULTS_V2_FILE,
    HYBRID_MODEL_V2_FILE,
)

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

# Data Files
HYBRID_ISIC = HYBRID_ISIC_FILE
META_ISIC = ISIC_METADATA_FILE
HYBRID_HAM = HYBRID_HAM_FILE
META_HAM = HAM_METADATA_FILE

RANDOM_SEED = 42

def evaluate_with_threshold(model, X, y, threshold=0.5):
    probs = model.predict_proba(X)
    if probs.shape[1] == 2:
        y_prob = probs[:, 1]
    else:
        y_prob = probs[:, 0]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ba": balanced_accuracy_score(y, y_pred),
        "sens": recall_score(y, y_pred),
        "spec": recall_score(y, y_pred, pos_label=0),
        "prec": precision_score(y, y_pred, zero_division=0),
        "acc": accuracy_score(y, y_pred),
        "cm": confusion_matrix(y, y_pred).tolist()
    }

def run_optimized_pipeline():
    print("Loading Hybrid Features & Metadata...")
    df_feats = pd.read_csv(HYBRID_ISIC)
    df_meta = pd.read_csv(META_ISIC)
    df = pd.merge(df_feats, df_meta, on='image_id', how='inner')
    print(f"Total Combined Dataset: {df.shape}")
    
    # 1. Feature Selection & PCA
    print("Preprocessing features...")
    df = df.drop(columns=['lesion_id', 'dx', 'image', 'anatom_site_general', 'localization'], errors='ignore')
    
    deep_cols = [f"deep_{i}" for i in range(2048)]
    other_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    hand_meta_cols = [c for c in other_numeric if c not in deep_cols and c != 'label']
    
    X_deep = df[deep_cols].values
    X_hand_meta = df[hand_meta_cols].values
    y = df['label'].values
    
    # 2. PCA for Deep Features
    print(f"Applying PCA to {len(deep_cols)} deep features...")
    pca = PCA(n_components=256, random_state=RANDOM_SEED)
    X_deep_pca = pca.fit_transform(X_deep)
    
    X = np.hstack([X_deep_pca, X_hand_meta])
    print(f"Final Feature Set Size: {X.shape[1]} (256 PCA + {len(hand_meta_cols)} Hand/Meta)")
    
    # 3. Normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # 4. Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED)
    
    # 5. Balancing
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    
    # 6. Train
    print("\nTraining ExSTraCS (500k iterations)...")
    model = ExSTraCS(learning_iterations=500000, N=3000, nu=10, track_accuracy_while_fit=True)
    start = time.time()
    model.fit(X_train_res, y_train_res)
    print(f"Training finished in {(time.time()-start)/60:.1f} minutes.")
    
    # 7. Multi-Threshold Tuning
    print("\nTuning Thresholds...")
    best_ba = 0
    std_threshold = 0.5
    for t in np.linspace(0.1, 0.9, 81):
        res = evaluate_with_threshold(model, X_val, y_val, threshold=t)
        if res['ba'] > best_ba:
            best_ba = res['ba']
            std_threshold = t
            
    # Safety Threshold (Target Sens >= 0.85)
    safety_threshold = std_threshold
    for t in np.linspace(std_threshold, 0.05, 50):
        res = evaluate_with_threshold(model, X_val, y_val, threshold=t)
        if res['sens'] >= 0.85:
            safety_threshold = t
            break
            
    print(f"Standard Threshold: {std_threshold:.2f} (Val BA: {best_ba:.4f})")
    print(f"Safety Threshold:   {safety_threshold:.2f}")

    # 8. Comparison Evaluation
    results = {'std': {}, 'safety': {}}
    for t_name, t_val in [('std', std_threshold), ('safety', safety_threshold)]:
        results[t_name]['train'] = evaluate_with_threshold(model, X_train, y_train, t_val)
        results[t_name]['val'] = evaluate_with_threshold(model, X_val, y_val, t_val)
        results[t_name]['test'] = evaluate_with_threshold(model, X_test, y_test, t_val)
    
    # 9. External Validation (HAM10000)
    if HYBRID_HAM.exists() and META_HAM.exists():
        print("\nRunning External Validation (HAM10000)...")
        df_ham_f = pd.read_csv(HYBRID_HAM)
        df_ham_m = pd.read_csv(META_HAM)
        df_ham = pd.merge(df_ham_f, df_ham_m, on='image_id', how='inner')
        
        # Align columns
        for col in hand_meta_cols:
            if col not in df_ham.columns: df_ham[col] = 0
            
        X_ham_deep = df_ham[deep_cols].values
        X_ham_hm = df_ham[hand_meta_cols].values
        
        # Apply PCA and Scaling
        X_ham_pca = pca.transform(X_ham_deep)
        X_ham = np.hstack([X_ham_pca, X_ham_hm])
        X_ham = scaler.transform(X_ham)
        y_ham = df_ham['label'].values
        
        for t_name, t_val in [('std', std_threshold), ('safety', safety_threshold)]:
            results[t_name]['external'] = evaluate_with_threshold(model, X_ham, y_ham, t_val)
            
    # Save Model & Metrics
    with open(HYBRID_RESULTS_V2_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    # Save PCA and Scaler inside the model object for unified pickling
    model.pca = pca
    model.scaler = scaler
    model.feature_cols = hand_meta_cols # Keep track of which cols we used
    model.pickle_model(str(HYBRID_MODEL_V2_FILE))
    print("\nDone. Results saved to results_hybrid_v2.json")

if __name__ == "__main__":
    run_optimized_pipeline()
