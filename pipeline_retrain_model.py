
"""
Pipeline Phase 2: Retrain Model with Deep Features
--------------------------------------------------
Trains ExSTraCS on the Top 200 ResNet50 features selected in Phase 1.
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
from project_paths import (
    EXSTRACS_LIB_DIR,
    EXSTRACS_VIZ_PARENT,
    DEEP_FEATURES_FILE,
    FEATURE_SELECTION_METADATA_FILE,
    DEEP_CHAMPION_MODEL_FILE,
    CUSTOM_LABELS_FILE,
    FINAL_METRICS_FILE,
    TRAINING_TRACKING_FILE,
    PLOTS_DIR,
)

# Setup Paths for independent execution
SCRIPT_DIR = Path(__file__).parent.resolve()
# Point to local library COPY
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

# Config
FEATURE_FILE = DEEP_FEATURES_FILE
METADATA_FILE = FEATURE_SELECTION_METADATA_FILE
MODEL_FILE = DEEP_CHAMPION_MODEL_FILE
LABEL_SOURCE_PATH = CUSTOM_LABELS_FILE
RANDOM_SEED = 42

def run_training_pipeline():
    # 1. Load Metadata
    if not METADATA_FILE.exists():
        print(f"Error: {METADATA_FILE} missing. Run Phase 1 first.")
        return
        
    with open(METADATA_FILE) as f:
        meta = json.load(f)
        top_indices = meta['top_indices']
        feature_names = meta['base_features'] # All names in original order
        
    # 2. Reconstruct Dataset
    print("Loading Data...")
    df_feats = pd.read_csv(FEATURE_FILE)
    df_labels = pd.read_csv(LABEL_SOURCE_PATH)
    df_merged = pd.merge(df_feats, df_labels[['image', 'label']], left_on='image_id', right_on='image', how='inner')
    
    from sklearn.utils import shuffle
    from sklearn.preprocessing import MinMaxScaler

    # 3. Select Features
    print(f"Aligning to Top {len(top_indices)} Deep Features...")
    # Get all potential feature columns first
    drop_cols = ['image_id', 'image', 'label']
    valid_cols = [c for c in df_merged.columns if c not in drop_cols]
    
    # Extract X (Full width)
    X_full = df_merged[valid_cols].values
    y = df_merged['label'].values
    
    # Filter columns to top selection
    X_selected = X_full[:, top_indices]
    
    # --- NORMALIZATION ---
    # Critical for ExSTraCS to work in [0,1] hyper-rectangles
    print("Applying Normalization (MinMaxScaler)...")
    scaler = MinMaxScaler()
    X_selected = scaler.fit_transform(X_selected)
    
    # 4. Split (Exact Replication of Selection Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    print(f"Training Set: {X_train.shape}")
    print(f"Test Set:     {X_test.shape}")
    
    # 5. Initialize ExSTraCS
    print("\nInitializing Independent ExSTraCS Model...")
    # Config mirroring Champion: N=3000, nu=10
    model = ExSTraCS(
        learning_iterations=500000, # Full run
        N=3000,
        nu=10,
        chi=0.8,
        mu=0.04,
        track_accuracy_while_fit=True
    )
    
    # 6. Train
    print("--- STARTING TRAINING (500k Iterations) ---")
    start = time.time()
    try:
        model.fit(X_train, y_train)
    except KeyboardInterrupt:
        print("\nTraining Interrupted! Saving Checkpoint...")
    except Exception as e:
        print(f"\nError during training: {e}")
        return
        
    duration = (time.time() - start) / 60
    print(f"Training finished in {duration:.1f} minutes.")
    
    # 7. Evaluate
    from sklearn.metrics import average_precision_score
    
    print("\nEvaluation (Internal Test Split):")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) # [n_samples, n_classes]
    
    # Handle Probabilities for Binary Case
    if y_proba.shape[1] == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba[:, 0]
    
    # Metrics
    ba = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    sens = tp / (tp + fn) # Recall (Malignant)
    spec = tn / (tn + fp) # Specificity (Benign)
    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    auc_pr = average_precision_score(y_test, y_score)
    
    print(f"  Balanced Accuracy: {ba:.4f}")
    print(f"  Sensitivity (Recall): {sens:.4f}")
    print(f"  Specificity:       {spec:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  AUC-PR:            {auc_pr:.4f}")
    print(f"  Overall Accuracy:  {acc:.4f}")
    
    # Save Metrics to JSON for Report
    metrics = {
        "balanced_accuracy": ba,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "auc_pr": auc_pr,
        "accuracy": acc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    with open(FINAL_METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 8. Save
    model.pickle_model(str(MODEL_FILE))
    print(f"Model saved to {MODEL_FILE}")
    
    # Export Tracking
    TRACKING_CSV = TRAINING_TRACKING_FILE
    model.export_iteration_tracking_data(str(TRACKING_CSV))

    # --- 9. VISUALIZATIONS ---
    print("\nGenerating Visualizations...")
    PLOTS_DIR.mkdir(exist_ok=True)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    import subprocess
    import os

    # A. Training Dynamics (exstracs_viz)
    # Ensure exstracs_viz is in path (it is in SCRIPT_DIR)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EXSTRACS_VIZ_PARENT) + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        print("  - Running exstracs_viz CLI...")
        subprocess.run([
            sys.executable, "-m", "exstracs_viz.cli",
            str(TRACKING_CSV),
            "--out", str(PLOTS_DIR)
        ], env=env, check=False)
    except Exception as e:
        print(f"Warning: exstracs_viz failed: {e}")

    # B. ROC & Precision-Recall Curves
    def plot_curves(y_true, y_pred_prob, title, filename):
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)
        
        plt.figure(figsize=(10, 5))
        
        # ROC
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f'ROC - {title}')
        plt.legend()
        
        # PRC
        plt.subplot(1, 2, 2)
        plt.plot(rec, prec, label=f'PRC (AUC={auc(rec, prec):.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f'PRC - {title}')
        plt.legend()
        
        plt.tight_layout()
    # C. Attribute Importance (Specificity)
    try:
        attr_spec = model.get_final_attribute_specificity_list()
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(attr_spec)), attr_spec)
        plt.title("Deep Feature Specificity (Attribute Importance)")
        plt.xlabel("Feature Index (in Top Selection)")
        plt.ylabel("Specificity")
        plt.savefig(PLOTS_DIR / "attribute_importance.png")
        plt.close()
        print("  - Importance plot saved.")
    except Exception as e:
        print(f"Warning: Could not plot importance: {e}")

    # D. Overfitting Audit (Train vs Test Accuracy Gap)
    try:
        df_track = pd.read_csv(TRACKING_CSV)
        # Check if we have both train and test accuracy columns
        # ExSTraCS usually tracks 'Training Accuracy' and 'Accuracy' (which is test/val if configured)
        # We need to check column names. Standard is 'Accuracy' (Training) and 'Validation Accuracy' (if Validation set provided)
        # But here we didn't provide explicit validation set to fit(), we rely on track_accuracy_while_fit which usually just tracks Training Accuracy.
        # Wait, if we only track training accuracy, we can't plot the gap over time unless we modified ExSTraCS to valid on test set every N iterations.
        # Standard ExSTraCS `track_accuracy_while_fit` calculates accuracy on the TRAINING set every 'accuracy_tracking_interval'.
        
        # NOTE: Without modifying ExSTraCS core, we can only plot Training Accuracy evolution.
        # But we can plot the FINAL Gap manually.
        # Or... does IterationData.csv contain "Validation Accuracy"? Only if we passed X_val to fit.
        # Let's check if we can pass X_test as validation to fit?
        # Looking at standard ExSTraCS, fit(X, y) doesn't take X_val.
        
        # Ideally, we plot Training Accuracy curve. And we annotate final Test Accuracy.
        
        if 'Accuracy' in df_track.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df_track['Iteration'], df_track['Accuracy'], label='Training Accuracy')
            # Add horizontal line for Final Test Accuracy
            plt.axhline(y=acc, color='r', linestyle='--', label=f'Final Test Acc ({acc:.2f})')
            plt.title("Overfitting Audit: Training Trajectory vs Final Test Performance")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.legend()
            
            # Gap annotation
            final_train_acc = df_track['Accuracy'].iloc[-1]
            gap = final_train_acc - acc
            plt.figtext(0.5, 0.01, f"Final Train-Test Gap: {gap*100:.1f}% (Target: < 2%)", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2})
            
            plt.savefig(PLOTS_DIR / "overfitting_audit.png")
            plt.close()
            print("  - Overfitting Audit plot saved.")
            
    except Exception as e:
        print(f"Warning: Could not plot overfitting audit: {e}")

if __name__ == "__main__":
    run_training_pipeline()

