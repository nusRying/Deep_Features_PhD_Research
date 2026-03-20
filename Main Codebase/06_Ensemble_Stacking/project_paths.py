from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

DATA_DIR = SCRIPT_DIR.parent / "data"
ROOT_DIR = SCRIPT_DIR
LABELS_DIR = DATA_DIR / "labels"
FEATURES_DIR = DATA_DIR / "features"
METADATA_DIR = DATA_DIR / "metadata"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "Plots"

EXTERNAL_DIR = ROOT_DIR / "external"
TOOLS_DIR = ROOT_DIR / "tools"

EXSTRACS_LIB_DIR = EXTERNAL_DIR / "scikit-ExSTraCS-master"
EXSTRACS_VIZ_PARENT = TOOLS_DIR

CUSTOM_LABELS_FILE = LABELS_DIR / "custom_labels.csv"
DEEP_FEATURES_FILE = FEATURES_DIR / "deep_features.csv"

HYBRID_ISIC_FILE = FEATURES_DIR / "hybrid_features_isic.csv"
HYBRID_HAM_FILE = FEATURES_DIR / "hybrid_features_ham.csv"
TEST_HYBRID_ISIC_FILE = FEATURES_DIR / "test_hybrid_isic.csv"
TEST_HYBRID_HAM_FILE = FEATURES_DIR / "test_hybrid_ham.csv"

ISIC_METADATA_FILE = METADATA_DIR / "metadata_isic_clean.csv"
HAM_METADATA_FILE = METADATA_DIR / "metadata_ham_clean.csv"
FEATURE_SELECTION_METADATA_FILE = METADATA_DIR / "feature_metadata.json"

DEEP_CHAMPION_MODEL_FILE = MODELS_DIR / "deep_champion_model.pkl"
ENSEMBLE_CHAMPION_FILE = MODELS_DIR / "ensemble_champion.pkl"
ENSEMBLE_CHECKPOINT_FILE = MODELS_DIR / "ensemble_champion_checkpoint.pkl"
HYBRID_MODEL_FILE = MODELS_DIR / "hybrid_model.pkl"
HYBRID_MODEL_V2_FILE = MODELS_DIR / "hybrid_model_v2.pkl"

FINAL_METRICS_FILE = RESULTS_DIR / "final_metrics.json"
HYBRID_RESULTS_FILE = RESULTS_DIR / "results_hybrid.json"
HYBRID_RESULTS_V2_FILE = RESULTS_DIR / "results_hybrid_v2.json"
HYBRID_IMPORTANCE_FILE = RESULTS_DIR / "hybrid_feature_importance.csv"
HYBRID_TOP_RULES_FILE = RESULTS_DIR / "hybrid_top_rules.txt"
TRAINING_TRACKING_FILE = RESULTS_DIR / "training_tracking.csv"
DIAGNOSTIC_REPORT_FILE = RESULTS_DIR / "diagnostic_report.json"
