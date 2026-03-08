import pandas as pd
import numpy as np
import time
import sys
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTENC
from project_paths import (
    EXSTRACS_LIB_DIR,
    HYBRID_ISIC_FILE,
    ISIC_METADATA_FILE,
    HYBRID_HAM_FILE,
    HAM_METADATA_FILE,
    TEST_HYBRID_ISIC_FILE,
    ENSEMBLE_CHECKPOINT_FILE,
    ENSEMBLE_CHAMPION_FILE,
)
# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

# Paths
HYBRID_ISIC = HYBRID_ISIC_FILE
META_ISIC = ISIC_METADATA_FILE
HYBRID_HAM = HYBRID_HAM_FILE
META_HAM = HAM_METADATA_FILE

RANDOM_SEED = 42

def evaluate_model(y_true, y_pred, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred) # Sensitivity for class 1
    spec = recall_score(y_true, y_pred, pos_label=0) # Specificity for class 0
    
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Acc: {ba:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    
    return {"acc": acc, "ba": ba, "sens": sens, "spec": spec}

def run_ensemble_pipeline(quick_test=False):
    print(f"Loading Data (Quick Test: {quick_test})...")
    feat_file = TEST_HYBRID_ISIC_FILE if quick_test and TEST_HYBRID_ISIC_FILE.exists() else HYBRID_ISIC
    df_feats = pd.read_csv(feat_file)
    df_meta = pd.read_csv(META_ISIC)
    df = pd.merge(df_feats, df_meta, on='image_id', how='inner')
    
    if quick_test:
        df = df.sample(n=min(500, len(df)), random_state=RANDOM_SEED)
    
    # Preprocessing
    df = df.drop(columns=['lesion_id', 'dx', 'image', 'anatom_site_general', 'localization'], errors='ignore')
    
    # Feature Categorization
    deep_cols = [f"deep_{i}" for i in range(2048)]
    hand_cols = [f"hand_{i}" for i in range(32)]
    
    # Metadata columns from CSV view
    meta_cols = [
        'age', 'sex_female', 'sex_male', 'sex_unknown',
        'site_head_neck', 'site_lower_extremity', 'site_torso', 'site_unknown', 'site_upper_extremity'
    ]
    
    # 0-8: Color, 9-15: Geometry, 16-21: GLCM, 22-31: LBP
    color_cols = [f"hand_{i}" for i in range(9)]
    geometry_cols = [f"hand_{i}" for i in range(9, 16)]
    texture_cols = [f"hand_{i}" for i in range(16, 32)]
    
    y = df['label'].values
    image_ids = df['image_id'].values

    # 1. Feature Reduction for Deep Features (Expert 1)
    print("Performing PCA on deep features...")
    n_samp = len(df)
    n_pca = min(256, n_samp - 1 if n_samp > 1 else 1)
    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    deep_pca = pca.fit_transform(df[deep_cols].values)
    deep_pca_cols = [f"pca_{i}" for i in range(n_pca)]
    df_deep_pca = pd.DataFrame(deep_pca, columns=deep_pca_cols, index=df.index)

    # expert 1 data
    X_exp1 = pd.concat([df_deep_pca, df[texture_cols]], axis=1)
    # expert 2 data
    X_exp2 = df[geometry_cols]
    # expert 3 data
    X_exp3 = df[meta_cols + color_cols]

    # Combine for splitting
    X_all = pd.concat([X_exp1, X_exp2, X_exp3], axis=1)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    
    # Identify indices for metadata (categorical) in X_all
    # Exp3 starts at 272 + 7 = 279 in the FULL set if PCA=256
    # But PCA is dynamic here. Let's find index by name.
    current_cols = X_train.columns.tolist()
    cat_indices = [i for i, col in enumerate(current_cols) if col in meta_cols and col != 'age']

    print("Applying balancing...")
    # Only use SMOTENC if we have at least 2 samples of each class and enough for cat indices
    if len(np.unique(y_train)) > 1 and len(y_train) > len(cat_indices):
        try:
            smote = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_SEED)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"Balanced Training Set: {len(X_train_res)} samples")
        except Exception as e:
            print(f"SMOTE-NC skipped due to: {e}")
            X_train_res, y_train_res = X_train, y_train
    else:
        print("Skipping SMOTE-NC (not enough samples for balancing)")
        X_train_res, y_train_res = X_train, y_train

    # Scalers (Independent for each expert)
    scaler1 = MinMaxScaler()
    X1_tr = scaler1.fit_transform(X_train_res[X_exp1.columns])
    X1_ts = scaler1.transform(X_test[X_exp1.columns])

    scaler2 = MinMaxScaler()
    X2_tr = scaler2.fit_transform(X_train_res[X_exp2.columns])
    X2_ts = scaler2.transform(X_test[X_exp2.columns])

    scaler3 = MinMaxScaler()
    X3_tr = scaler3.fit_transform(X_train_res[X_exp3.columns])
    X3_ts = scaler3.transform(X_test[X_exp3.columns])

    iters = 5000 if quick_test else 500000
    
    # Train Expert 1 (Visual)
    print("\nTraining Expert 1 (Visual/Deep)...")
    exp1 = ExSTraCS(learning_iterations=iters, N=3000, nu=10)
    exp1.fit(X1_tr, y_train_res)

    # Train Expert 2 (Geometry)
    print("\nTraining Expert 2 (Geometry)...")
    exp2 = ExSTraCS(learning_iterations=iters, N=2000, nu=5)
    exp2.fit(X2_tr, y_train_res)

    # Train Expert 3 (Contextual)
    print("\nTraining Expert 3 (Contextual)...")
    exp3 = ExSTraCS(learning_iterations=iters, N=2000, nu=5)
    exp3.fit(X3_tr, y_train_res)

    # Interim Save - Important to avoid data loss on crash
    print("\nSaving Checkpoint Models...")
    import pickle
    with open(ENSEMBLE_CHECKPOINT_FILE, "wb") as f:
        pickle.dump({
            "exp1": exp1, "scaler1": scaler1,
            "exp2": exp2, "scaler2": scaler2,
            "exp3": exp3, "scaler3": scaler3,
            "pca": pca
        }, f)

    # Prediction (Soft Voting)
    print("\nEvaluating Ensemble...")
    prob1 = exp1.predict_proba(X1_ts)[:, 1]
    prob2 = exp2.predict_proba(X2_ts)[:, 1]
    prob3 = exp3.predict_proba(X3_ts)[:, 1]

    # Weighted Soft Voting (Weights can be tuned, starting with equal)
    ensemble_prob = (prob1 + prob2 + prob3) / 3
    y_pred = (ensemble_prob >= 0.5).astype(int)

    results = evaluate_model(y_test, y_pred, name="Grand Champion Ensemble (ISIC Internal)")
    
    # External Validation on HAM10000
    print("\nStarting External Validation on HAM10000...")
    df_ham_feats = pd.read_csv(HYBRID_HAM)
    df_ham_meta = pd.read_csv(META_HAM)
    df_ham = pd.merge(df_ham_feats, df_ham_meta, on='image_id', how='inner')
    
    # Feature Categorization for HAM
    X_ham_deep = df_ham[deep_cols].values
    X_ham_pca = pca.transform(X_ham_deep)
    df_ham_pca = pd.DataFrame(X_ham_pca, columns=deep_pca_cols, index=df_ham.index)
    
    X1_ham = scaler1.transform(pd.concat([df_ham_pca, df_ham[texture_cols]], axis=1))
    X2_ham = scaler2.transform(df_ham[geometry_cols])
    X3_ham = scaler3.transform(df_ham[meta_cols + color_cols])
    
    y_ham = df_ham['label'].values
    
    prob1_ham = exp1.predict_proba(X1_ham)[:, 1]
    prob2_ham = exp2.predict_proba(X2_ham)[:, 1]
    prob3_ham = exp3.predict_proba(X3_ham)[:, 1]
    
    ensemble_prob_ham = (prob1_ham + prob2_ham + prob3_ham) / 3
    y_pred_ham = (ensemble_prob_ham >= 0.5).astype(int)
    
    res_ham = evaluate_model(y_ham, y_pred_ham, name="Grand Champion Ensemble (HAM External)")
    
    # Save Models
    import pickle
    with open(ENSEMBLE_CHAMPION_FILE, "wb") as f:
        pickle.dump({
            "exp1": exp1, "scaler1": scaler1,
            "exp2": exp2, "scaler2": scaler2,
            "exp3": exp3, "scaler3": scaler3,
            "pca": pca,
            "results_isic": results,
            "results_ham": res_ham
        }, f)

if __name__ == "__main__":
    quick = "--quick-test" in sys.argv
    run_ensemble_pipeline(quick_test=quick)
