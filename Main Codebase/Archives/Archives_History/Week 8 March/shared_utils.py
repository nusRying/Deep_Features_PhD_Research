import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, recall_score, matthews_corrcoef
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

def load_clinical_data(data_dir, n_pca=256):
    """Unified loader for ISIC + HAM10000 data with PCA reduction."""
    logger.info(f"Loading data from {data_dir}...")
    
    # Files
    feat_file = Path(data_dir) / "features" / "hybrid_features_isic.csv"
    meta_file = Path(data_dir) / "metadata" / "metadata_isic_clean.csv"
    
    if not feat_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    df_isic = pd.merge(pd.read_csv(feat_file), pd.read_csv(meta_file), on='image_id')
    
    # Feature extraction logic
    deep_cols = [f"deep_{i}" for i in range(2048)]
    hand_cols = [c for c in df_isic.columns if c.startswith('hand_')]
    meta_cols = ['age']
    
    X_deep = df_isic[deep_cols].values
    X_clinical = df_isic[hand_cols + meta_cols].values
    y = df_isic['label'].values
    
    # PCA on deep features only (Preserving metadata/handcrafted integrity)
    pca = PCA(n_components=n_pca, random_state=42)
    X_deep_red = pca.fit_transform(X_deep)
    
    # Re-merge
    X_combined = np.hstack([X_deep_red, X_clinical])
    
    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Feature Names
    feature_names = [f"PCA_{i}" for i in range(n_pca)] + hand_cols + meta_cols
    
    return X_scaled, y, feature_names

def compute_phd_metrics(y_true, y_pred, y_probs=None):
    """Standardized performance auditing for thesis results."""
    metrics = {
        'ba': balanced_accuracy_score(y_true, y_pred),
        'sens': recall_score(y_true, y_pred),
        'spec': recall_score(y_true, y_pred, pos_label=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_probs is not None:
        # Expected Calibration Error (ECE) Simplified
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
        metrics['ece'] = np.mean(np.abs(prob_true - prob_pred))
        
    return metrics
