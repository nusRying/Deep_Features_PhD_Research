import sys
import json
import numpy as np
import pandas as pd
import pickle
import logging
import argparse
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, accuracy_score, 
    recall_score, precision_score, roc_curve, auc, precision_recall_curve,
    matthews_corrcoef, brier_score_loss
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.utils import resample

# Setup Standalone Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
LOG_DIR = SCRIPT_DIR / "logs"
EXTERNAL_DIR = SCRIPT_DIR / "external"
sys.path.insert(0, str(EXTERNAL_DIR / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

# Data Files
HYBRID_ISIC_FILE = DATA_DIR / "features" / "hybrid_features_isic.csv"
HYBRID_HAM_FILE = DATA_DIR / "features" / "hybrid_features_ham.csv"
ISIC_METADATA_FILE = DATA_DIR / "metadata" / "metadata_isic_clean.csv"
HAM_METADATA_FILE = DATA_DIR / "metadata" / "metadata_ham_clean.csv"

# Output Files
RESULTS_FILE = RESULTS_DIR / "results_v7_knowledge.json"
MODEL_FILE = MODELS_DIR / "hybrid_ensemble_v7_knowledge.pkl"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints_v7"
ATTR_SUMS_FILE = SCRIPT_DIR / "attr_tracking_sums.npy"
REPORT_FILE = RESULTS_DIR / "clinical_consensus_report.txt"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "v7_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# PhD Meta-Parameters
SEEDS = [42, 123, 789, 2024, 99]
LEARNING_ITERATIONS = 500000 
N_POP = 5000
BOOTSTRAP_SAMPLES = 1000

def extract_rules_from_ensemble(models_tier1, feature_names):
    """Extracts human-readable rules from the LCS experts."""
    master_rules = []
    logger.info("Extracting rules from Tier 1 experts...")
    for idx, model in enumerate(models_tier1):
        try:
            # model is ExSTraCS instance
            population = model.expert.population
            for cl in population:
                rule_str = cl.get_rule_string(feature_names)
                master_rules.append({
                    'expert': idx,
                    'rule': rule_str,
                    'fitness': cl.fitness,
                    'accuracy': cl.accuracy,
                    'prediction': cl.prediction
                })
        except Exception as e:
            logger.warning(f"Could not extract rules from expert {idx}: {e}")
    return master_rules

def generate_clinical_report(rules):
    """Generates a consensus report of high-fitness clinical rules."""
    logger.info("Generating Clinical Consensus Report...")
    df_rules = pd.DataFrame(rules)
    if df_rules.empty: return "No rules discovered or ExSTraCS version incompatible with direct extraction."
    
    top_rules = df_rules.sort_values('fitness', ascending=False).head(50)
    
    report = ["--- CLINICAL CONSENSUS REPORT: LESION DIAGNOSTIC LOGIC ---",
              f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"Analysis based on 5-fold Ensemble Ruleset",
              "---------------------------------------------------------"]
    
    for i, (idx, row) in enumerate(top_rules.iterrows()):
        outcome = "MALIGNANT" if row['prediction'] == 1 else "BENIGN"
        report.append(f"Knowledge Item {i+1}:")
        report.append(f"  Logic: {row['rule']}")
        report.append(f"  Confidence Profile: Prediction={outcome}, Fitness={row['fitness']:.4f}, Accuracy={row['accuracy']:.2f}")
        report.append("")
        
    return "\n".join(report)

def calculate_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals[non_empty_bins] / len(y_prob)
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return float(ece)

def get_metrics_comprehensive(y_true, y_prob, threshold=0.5):
    def calc_metrics(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        return {
            "ba": balanced_accuracy_score(y_t, y_pred),
            "sens": recall_score(y_t, y_pred),
            "spec": recall_score(y_t, y_pred, pos_label=0),
            "mcc": matthews_corrcoef(y_t, y_pred),
            "brier": brier_score_loss(y_t, y_p),
            "ece": calculate_ece(y_t, y_p)
        }
    base_metrics = calc_metrics(y_true, y_prob)
    boot_stats = {k: [] for k in base_metrics.keys()}
    for i in range(BOOTSTRAP_SAMPLES):
        indices = resample(np.arange(len(y_true)), replace=True)
        res = calc_metrics(y_true[indices], y_prob[indices])
        for k in boot_stats.keys(): boot_stats[k].append(res[k])
    
    final_results = {}
    for k in base_metrics.keys():
        final_results[k] = {
            "val": float(base_metrics[k]),
            "ci_lower": float(np.percentile(boot_stats[k], 2.5)),
            "ci_upper": float(np.percentile(boot_stats[k], 97.5))
        }
    return final_results

def run_v7_knowledge(smoke_test=False):
    global BOOTSTRAP_SAMPLES, LEARNING_ITERATIONS, N_POP
    if smoke_test:
        BOOTSTRAP_SAMPLES, LEARNING_ITERATIONS, N_POP = 5, 2000, 500
    
    logger.info("--- PHASE 7: KNOWLEDGE DISCOVERY (V7) ---")
    
    # Load
    df_isic = pd.merge(pd.read_csv(HYBRID_ISIC_FILE), pd.read_csv(ISIC_METADATA_FILE), on='image_id')
    deep_cols = [f"deep_{i}" for i in range(2048)]
    hand_meta_cols = [c for c in df_isic.select_dtypes(include=[np.number]).columns if c not in deep_cols and c != 'label']
    X_deep = df_isic[deep_cols].values
    y = df_isic['label'].values
    
    # PCA
    pca = PCA(n_components=256, random_state=42)
    X_raw = np.hstack([pca.fit_transform(X_deep), df_isic[hand_meta_cols].values])
    all_feature_names = [f"PCA_{i}" for i in range(256)] + list(hand_meta_cols)
    
    # LCS-Pruning
    if ATTR_SUMS_FILE.exists():
        attr_sums = np.load(ATTR_SUMS_FILE)
        keep_indices = np.where(attr_sums > np.median(attr_sums))[0]
        X_pruned = X_raw[:, keep_indices]
        pruned_feature_names = [all_feature_names[i] for i in keep_indices]
    else:
        X_pruned, pruned_feature_names = X_raw, all_feature_names

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_pruned)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # HAM10000 
    df_ham = pd.merge(pd.read_csv(HYBRID_HAM_FILE), pd.read_csv(HAM_METADATA_FILE), on='image_id')
    for c in hand_meta_cols: 
        if c not in df_ham.columns: df_ham[c] = 0
    X_ham = scaler.transform(np.hstack([pca.transform(df_ham[deep_cols].values), df_ham[hand_meta_cols].values])[:, keep_indices if ATTR_SUMS_FILE.exists() else np.arange(X_raw.shape[1])])
    y_ham = df_ham['label'].values

    # Tier 1: Experts (OOF)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.zeros(len(y_train))
    models_tier1 = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"Fold {fold+1}/5 Training...")
        ckpt = CHECKPOINT_DIR / f"expert_fold_{fold}.pkl"
        if ckpt.exists() and not smoke_test:
            with open(ckpt, 'rb') as f: model = pickle.load(f)
        else:
            X_res, y_res = SMOTEENN(random_state=42).fit_resample(X_train[t_idx], y_train[t_idx])
            model = ExSTraCS(learning_iterations=LEARNING_ITERATIONS, N=N_POP, nu=5)
            model.fit(X_res, y_res)
            with open(ckpt, 'wb') as f: pickle.dump(model, f)
        
        oof_meta[v_idx] = model.predict_proba(X_train[v_idx])[:, 1]
        models_tier1.append(model)
        if smoke_test and fold >= 0: break

    # Meta-Learners
    meta_a = LogisticRegression(class_weight={0: 1, 1: 2.0}).fit(oof_meta.reshape(-1, 1), y_train)
    meta_b = RandomForestClassifier(n_estimators=100, max_depth=3).fit(oof_meta.reshape(-1, 1), y_train)
    
    # Aggregation & Calibration
    def get_tier1_preds(X_in): return np.mean([m.predict_proba(X_in)[:, 1] for m in models_tier1], axis=0)
    p_val_agg = (meta_a.predict_proba(get_tier1_preds(X_val).reshape(-1, 1))[:, 1] * 0.7 + 
                 meta_b.predict_proba(get_tier1_preds(X_val).reshape(-1, 1))[:, 1] * 0.3)
    calibrator = IsotonicRegression(out_of_bounds='clip').fit(p_val_agg, y_val)
    
    # Final Inference
    def final_pred(X_in):
        p1 = get_tier1_preds(X_in).reshape(-1, 1)
        p_agg = meta_a.predict_proba(p1)[:, 1] * 0.7 + meta_b.predict_proba(p1)[:, 1] * 0.3
        return calibrator.transform(p_agg)

    # Knowledge Extraction
    rules = extract_rules_from_ensemble(models_tier1, pruned_feature_names)
    with open(REPORT_FILE, 'w') as f: f.write(generate_clinical_report(rules))
    
    # Results Evaluation
    p_val_final = calibrator.transform(p_val_agg)
    best_ba, best_t = 0, 0.5
    for t in np.linspace(0.01, 0.99, 100):
        ba = balanced_accuracy_score(y_val, (p_val_final >= t).astype(int))
        if ba > best_ba: best_ba, best_t = ba, t

    results = {
        'v7_knowledge': {
            'internal': get_metrics_comprehensive(y_test, final_pred(X_test), best_t),
            'external': get_metrics_comprehensive(y_ham, final_pred(X_ham), best_t)
        }
    }
    with open(RESULTS_FILE, 'w') as f: json.dump(results, f, indent=4)
    logger.info(f"PHASE 7 COMPLETE. Results: {results['v7_knowledge']['internal']['ba']['val']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_v7_knowledge(smoke_test=args.smoke_test)
