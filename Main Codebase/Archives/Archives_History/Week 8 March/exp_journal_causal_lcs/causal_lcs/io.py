import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CausalExperimentConfig
from .metadata import CausalMetadataBuilder


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data"


def get_dataset_paths(config):
    if config.train_dataset.lower() == "ham":
        return {
            "features": DATA_DIR / "features" / "hybrid_features_ham.csv",
            "metadata": DATA_DIR / "metadata" / "metadata_ham_clean.csv",
        }
    return {
        "features": DATA_DIR / "features" / "hybrid_features_isic.csv",
        "metadata": DATA_DIR / "metadata" / "metadata_isic_clean.csv",
    }


def load_config(config_path=None):
    if config_path is None:
        return CausalExperimentConfig()
    config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return CausalExperimentConfig(**config_data)


def load_training_frame(config):
    dataset_paths = get_dataset_paths(config)
    feature_frame = pd.read_csv(dataset_paths["features"])
    metadata_frame = pd.read_csv(dataset_paths["metadata"])

    if config.label_column in feature_frame.columns:
        return feature_frame

    if config.metadata_join_key not in feature_frame.columns or config.metadata_join_key not in metadata_frame.columns:
        raise ValueError("Expected join key is missing from one or both input tables.")

    merged = pd.merge(feature_frame, metadata_frame, on=config.metadata_join_key, how="inner")
    if config.label_column not in merged.columns:
        raise ValueError("Merged training frame does not contain the configured label column.")
    return merged


def select_feature_columns(frame, config):
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    excluded = set(config.excluded_feature_columns + [config.label_column])
    return [column for column in numeric_columns if column not in excluded]


def save_runtime_manifest(config, feature_columns, output_path):
    payload = {
        "config": asdict(config),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_design_matrix(frame, feature_columns, config):
    x_raw = frame[feature_columns].astype(float).to_numpy(dtype=np.float64)
    y = frame[config.label_column].astype(int).to_numpy()
    image_ids = frame[config.image_id_column].astype(str).tolist() if config.image_id_column in frame.columns else []
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_raw)
    return {
        "X": x_scaled,
        "y": y,
        "image_ids": image_ids,
        "scaler": scaler,
    }


def build_runtime_bundle(config_path=None):
    config = load_config(config_path)
    train_frame = load_training_frame(config)
    feature_columns = select_feature_columns(train_frame, config)
    metadata = CausalMetadataBuilder(feature_columns, config).build()
    design = prepare_design_matrix(train_frame, feature_columns, config)
    return {
        "config": config,
        "train_frame": train_frame,
        "feature_columns": feature_columns,
        "metadata": metadata,
        "X": design["X"],
        "y": design["y"],
        "image_ids": design["image_ids"],
        "scaler": design["scaler"],
    }