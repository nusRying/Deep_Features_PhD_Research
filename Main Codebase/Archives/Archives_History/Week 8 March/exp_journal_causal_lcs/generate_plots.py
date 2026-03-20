import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Paths
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
import sys
TOOLS_DIR = str((BASE_DIR.parent.parent / "tools" / "exstracs_viz").resolve())
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
import plotting
import pickle
import json

# Paths
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TRACKING_CSV = RESULTS_DIR / "tracking" / "iterationData.csv"
SUMMARY_JSON = RESULTS_DIR / "train_run_summary.json"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Training Dynamics Plots
print("Generating training dynamics plots...")
df_track = pd.read_csv(TRACKING_CSV)
plotting.plot_performance(df_track, save_path=PLOTS_DIR)
plotting.plot_population(df_track, save_path=PLOTS_DIR)
plotting.plot_sets(df_track, save_path=PLOTS_DIR)
plotting.plot_operations(df_track, save_path=PLOTS_DIR)
plotting.plot_timing(df_track, save_path=PLOTS_DIR)

# 2. ROC & PRC for Test Split
print("Generating ROC and PRC plots for test split...")
with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
    summary = json.load(f)

# If you want to plot from raw predictions, you need to load the model and test data.
# Here, we only plot from summary if available. For full ROC/PRC, you need y_test and y_pred_prob.
# If you want to add this, uncomment and adapt the following:
# from causal_lcs.workflow import create_data_splits
# from causal_lcs import build_runtime_bundle
# runtime = build_runtime_bundle()  # or load config if needed
# split_bundle = create_data_splits(runtime)
# x_test, y_test = split_bundle["test"]
# model = ... # load model if needed
# y_pred_prob = model.predict_proba(x_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
# prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
# ...

# For now, just print summary
print("Test split metrics:")
for k, v in summary["test"].items():
    print(f"  {k}: {v}")

print(f"All plots saved to: {PLOTS_DIR}")
