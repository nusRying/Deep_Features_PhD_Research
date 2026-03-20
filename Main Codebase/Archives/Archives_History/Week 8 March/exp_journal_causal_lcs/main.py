import argparse
from pathlib import Path
import sys

from causal_lcs import build_runtime_bundle, create_data_splits, save_split_summary, save_training_outputs, train_single_run
from causal_lcs.io import save_runtime_manifest

EXPERIMENT_DIR = Path(__file__).resolve().parent
LOCAL_EXSTRACS_ROOT = Path(__file__).resolve().parent / "external" / "scikit-ExSTraCS-master"
if str(LOCAL_EXSTRACS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_EXSTRACS_ROOT))

from skExSTraCS import ExSTraCS


def prepare_runtime(config_path=None):
    runtime = build_runtime_bundle(config_path)
    metadata = runtime["metadata"]
    artifact_dir = EXPERIMENT_DIR / "results" / "artifacts"
    metadata.save(artifact_dir / "causal_metadata.json")
    save_runtime_manifest(runtime["config"], runtime["feature_columns"], artifact_dir / "runtime_manifest.json")
    split_bundle = create_data_splits(runtime)
    save_split_summary(split_bundle, EXPERIMENT_DIR / "results" / "artifacts" / "split_summary.txt")
    return runtime


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Isolated Causal-LCS experiment entrypoint")
    parser.add_argument("--config", type=str, default=None, help="Optional path to a JSON config file")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare metadata and split artifacts only")
    parser.add_argument("--train", action="store_true", help="Train a single local Causal-LCS run")
    parser.add_argument("--learning-iterations", type=int, default=100000)
    parser.add_argument("--population-size", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    runtime = prepare_runtime(args.config)

    if args.train:
        run_result = train_single_run(
            ExSTraCS,
            runtime,
            learning_iterations=args.learning_iterations,
            population_size=args.population_size,
            random_state=args.random_state,
        )
        saved_outputs = save_training_outputs(run_result, EXPERIMENT_DIR)
        run_result["saved_outputs"] = saved_outputs

        # === Plotting Section ===
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        from tools.exstracs_viz import plotting

        results_dir = Path(saved_outputs["metrics_path"]).parent
        tracking_csv = saved_outputs["tracking_path"]
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Training Dynamics Plots
        df_track = pd.read_csv(tracking_csv)
        plotting.plot_performance(df_track, save_path=plots_dir)
        plotting.plot_population(df_track, save_path=plots_dir)
        plotting.plot_sets(df_track, save_path=plots_dir)
        plotting.plot_operations(df_track, save_path=plots_dir)
        plotting.plot_timing(df_track, save_path=plots_dir)

        # 2. ROC & PRC for Test Split
        # Get test split from runtime
        from causal_lcs.workflow import create_data_splits
        split_bundle = create_data_splits(runtime, random_state=args.random_state)
        x_test, y_test = split_bundle["test"]
        model = run_result["model"]
        y_pred_prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_test)
            if probs.shape[1] == 2:
                y_pred_prob = probs[:, 1]
            else:
                y_pred_prob = probs[:, 0]
        if y_pred_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
            plt.figure(figsize=(10, 5))
            # ROC
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f'ROC (AUC={auc(fpr, tpr):.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title('ROC - Test Split')
            plt.legend()
            # PRC
            plt.subplot(1, 2, 2)
            plt.plot(rec, prec, label=f'PRC (AUC={auc(rec, prec):.2f})')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title('PRC - Test Split')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "roc_prc_test.png")
            plt.close()

        print(f"All metrics and plots saved to: {results_dir}")
        return run_result

    return runtime


if __name__ == "__main__":
    main()