import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class CausalMetadata:
    dag_edges: List[List[str]] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)
    treatment_features: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    feature_index_map: Dict[str, int] = field(default_factory=dict)
    outcome_name: str = "phenotype"
    propensity_scores: Dict[str, float] = field(default_factory=dict)
    rule_parent_map: Dict[int, List[str]] = field(default_factory=dict)

    def save(self, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class CausalMetadataBuilder:
    def __init__(self, feature_names, config):
        self.feature_names = list(feature_names)
        self.config = config

    def build(self):
        confounders = self._select_features(self.config.confounders, self.config.confounder_prefixes)
        treatments = self._select_features(self.config.treatment_features, self.config.treatment_prefixes)
        feature_index_map = {feature_name: index for index, feature_name in enumerate(self.feature_names)}
        dag_edges = [[confounder, treatment] for confounder in confounders for treatment in treatments]
        dag_edges.extend([[feature, self.config.outcome_name] for feature in treatments])
        propensity_scores = {feature: 1.0 for feature in self.feature_names}
        rule_parent_map = self._build_rule_parent_map(feature_index_map, confounders, treatments)
        return CausalMetadata(
            dag_edges=dag_edges,
            confounders=confounders,
            treatment_features=treatments,
            feature_columns=list(self.feature_names),
            feature_index_map=feature_index_map,
            outcome_name=self.config.outcome_name,
            propensity_scores=propensity_scores,
            rule_parent_map=rule_parent_map,
        )

    def _select_features(self, explicit_names, prefixes):
        selected = []
        explicit_set = set(explicit_names)
        for feature_name in self.feature_names:
            if feature_name in explicit_set:
                selected.append(feature_name)
                continue
            for prefix in prefixes:
                if feature_name.startswith(prefix):
                    selected.append(feature_name)
                    break
        return selected

    def _build_rule_parent_map(self, feature_index_map, confounders, treatments):
        confounder_nodes = list(confounders)
        rule_parent_map = {}
        for feature_name in self.feature_names:
            feature_index = feature_index_map[feature_name]
            if feature_name in treatments:
                rule_parent_map[feature_index] = confounder_nodes + ["treatment"]
            elif feature_name in confounders:
                rule_parent_map[feature_index] = ["confounder"]
            else:
                rule_parent_map[feature_index] = []
        for key, value in self.config.rule_parent_map.items():
            rule_parent_map[int(key)] = list(value)
        return rule_parent_map