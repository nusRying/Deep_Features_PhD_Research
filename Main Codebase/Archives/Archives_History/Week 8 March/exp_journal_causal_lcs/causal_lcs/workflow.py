from pathlib import Path
import json
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .fitness import CausalFitnessPolicy
from .subsumption import CausalSubsumptionPolicy


def create_data_splits(runtime, random_state=42, validation_size=0.2, test_size=0.2):
    x_train, x_temp, y_train, y_temp = train_test_split(
        runtime["X"],
        runtime["y"],
        test_size=validation_size + test_size,
        stratify=runtime["y"],
        random_state=random_state,
    )
    relative_test_size = test_size / float(validation_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )
    return {
        "train": (x_train, y_train),
        "validation": (x_val, y_val),
        "test": (x_test, y_test),
    }


def instantiate_model(exstracs_class, runtime, learning_iterations=100000, population_size=5000, random_state=42):
    metadata = runtime["metadata"]
    return exstracs_class(
        learning_iterations=learning_iterations,
        N=population_size,
        random_state=random_state,
        fitness_policy=CausalFitnessPolicy(metadata, alpha=runtime["config"].alpha),
        subsumption_policy=CausalSubsumptionPolicy(metadata),
    )


def save_split_summary(split_bundle, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = []
    for split_name, (x_values, y_values) in split_bundle.items():
        positive_rate = float(np.mean(y_values)) if len(y_values) else 0.0
        summary_lines.append(f"{split_name}: n={len(y_values)}, positive_rate={positive_rate:.6f}, features={x_values.shape[1]}")
    path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def train_single_run(exstracs_class, runtime, learning_iterations=100000, population_size=5000, random_state=42):
    split_bundle = create_data_splits(runtime, random_state=random_state)
    model = instantiate_model(
        exstracs_class,
        runtime,
        learning_iterations=learning_iterations,
        population_size=population_size,
        random_state=random_state,
    )
    x_train, y_train = split_bundle["train"]
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, split_bundle)
    return {
        "model": model,
        "splits": split_bundle,
        "metrics": metrics,
    }


def evaluate_model(model, split_bundle):
    results = {}
    for split_name, (x_values, y_values) in split_bundle.items():
        predictions = model.predict(x_values)
        probabilities = model.predict_proba(x_values)
        split_metrics = {
            "accuracy": float(accuracy_score(y_values, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_values, predictions)),
            "sensitivity": float(recall_score(y_values, predictions, zero_division=0)),
            "specificity": float(recall_score(y_values, predictions, pos_label=0, zero_division=0)),
        }
        positive_probs = extract_positive_probabilities(probabilities)
        if positive_probs is not None:
            try:
                split_metrics["roc_auc"] = float(roc_auc_score(y_values, positive_probs))
            except Exception:
                split_metrics["roc_auc"] = None
        else:
            split_metrics["roc_auc"] = None
        results[split_name] = split_metrics
    return results


def extract_positive_probabilities(probabilities):
    if probabilities is None:
        return None
    if len(probabilities.shape) != 2:
        return None
    if probabilities.shape[1] < 2:
        return None
    return probabilities[:, 1]


def save_training_outputs(run_result, output_dir):
    output_root = Path(output_dir)
    models_dir = output_root / "models"
    results_dir = output_root / "results"
    tracking_dir = results_dir / "tracking"
    models_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "causal_lcs_model.pkl"
    metrics_path = results_dir / "train_run_summary.json"
    tracking_path = tracking_dir / "iterationData.csv"

    with open(model_path, "wb") as model_file:
        pickle.dump(run_result["model"], model_file)

    metrics_path.write_text(json.dumps(run_result["metrics"], indent=2), encoding="utf-8")
    run_result["model"].record.exportTrackingToCSV(str(tracking_path))

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "tracking_path": str(tracking_path),
    }