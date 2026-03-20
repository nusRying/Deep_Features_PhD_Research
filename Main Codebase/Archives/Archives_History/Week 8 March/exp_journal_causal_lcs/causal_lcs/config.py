from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CausalExperimentConfig:
    alpha: float = 0.7
    image_id_column: str = "image_id"
    label_column: str = "label"
    metadata_join_key: str = "image_id"
    treatment_features: List[str] = field(default_factory=list)
    treatment_prefixes: List[str] = field(default_factory=lambda: ["deep_", "hand_"])
    confounders: List[str] = field(default_factory=lambda: ["age"])
    confounder_prefixes: List[str] = field(default_factory=lambda: ["sex_", "site_"])
    excluded_feature_columns: List[str] = field(default_factory=lambda: ["image_id", "label"])
    outcome_name: str = "phenotype"
    rule_parent_map: Dict[int, List[str]] = field(default_factory=dict)
    propensity_column: str = "propensity_score"
    train_dataset: str = "isic"