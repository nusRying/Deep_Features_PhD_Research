import sys
import os
from pathlib import Path
# Ensure project_paths.py is found at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, recall_score
from project_paths import EXSTRACS_LIB_DIR, HYBRID_HAM_FILE, HAM_METADATA_FILE, ENSEMBLE_CHAMPION_FILE

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

# Paths
HYBRID_HAM = HYBRID_HAM_FILE
META_HAM = HAM_METADATA_FILE
MODEL_PATH = ENSEMBLE_CHAMPION_FILE

def evaluate_model(y_true, y_pred, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    spec = recall_score(y_true, y_pred, pos_label=0)
    
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Acc: {ba:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    
    return {"acc": acc, "ba": ba, "sens": sens, "spec": spec}

def run_external_validation():
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file {MODEL_PATH} not found. Training might still be in progress.")
        return

    print("Loading Trained Ensemble Model...")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    
    exp1 = data['exp1']
    exp2 = data['exp2']
    exp3 = data['exp3']
    scaler1 = data['scaler1']
    scaler2 = data['scaler2']
    scaler3 = data['scaler3']
    pca = data.get('pca')

    print("Loading HAM10000 External Dataset...")
    df_ham_feats = pd.read_csv(HYBRID_HAM)
    df_ham_meta = pd.read_csv(META_HAM)
    df_ham = pd.merge(df_ham_feats, df_ham_meta, on='image_id', how='inner')
    
    # Feature Categorization
    deep_cols = [f"deep_{i}" for i in range(2048)]
    texture_cols = [f"hand_{i}" for i in range(16, 32)]
    geometry_cols = [f"hand_{i}" for i in range(9, 16)]
    meta_cols = [
        'age', 'sex_female', 'sex_male', 'sex_unknown',
        'site_head_neck', 'site_lower_extremity', 'site_torso', 'site_unknown', 'site_upper_extremity'
    ]
    color_cols = [f"hand_{i}" for i in range(9)]

    print("Processing features for HAM...")
    X_ham_deep = df_ham[deep_cols].values
    X_ham_pca = pca.transform(X_ham_deep)
    df_ham_pca = pd.DataFrame(X_ham_pca, columns=[f"pca_{i}" for i in range(X_ham_pca.shape[1])], index=df_ham.index)
    
    X1_ham = scaler1.transform(pd.concat([df_ham_pca, df_ham[texture_cols]], axis=1))
    X2_ham = scaler2.transform(df_ham[geometry_cols])
    X3_ham = scaler3.transform(df_ham[meta_cols + color_cols])
    
    y_ham = df_ham['label'].values
    
    print("Generating Predictions (Ensemble Soft Voting)...")
    prob1_ham = exp1.predict_proba(X1_ham)[:, 1]
    prob2_ham = exp2.predict_proba(X2_ham)[:, 1]
    prob3_ham = exp3.predict_proba(X3_ham)[:, 1]
    
    ensemble_prob_ham = (prob1_ham + prob2_ham + prob3_ham) / 3
    y_pred_ham = (ensemble_prob_ham >= 0.5).astype(int)
    
    results = evaluate_model(y_ham, y_pred_ham, name="Grand Champion Ensemble (HAM External)")
    
    print("\nClassification Report:")
    print(classification_report(y_ham, y_pred_ham))

if __name__ == "__main__":
    run_external_validation()
