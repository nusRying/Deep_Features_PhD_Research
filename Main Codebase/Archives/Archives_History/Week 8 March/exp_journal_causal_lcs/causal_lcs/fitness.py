class CausalFitnessPolicy:
    def __init__(self, metadata, alpha=0.7):
        self.metadata = metadata
        self.alpha = alpha
        self.treatment_indices = {
            self.metadata.feature_index_map[feature_name]
            for feature_name in self.metadata.treatment_features
            if feature_name in self.metadata.feature_index_map
        }

    def calculate(self, model, classifier):
        predictive_score = pow(classifier.accuracy, model.nu)
        causal_score = self._estimate_rule_causal_score(classifier)
        return (self.alpha * causal_score) + ((1.0 - self.alpha) * predictive_score)

    def _estimate_rule_causal_score(self, classifier):
        if not classifier.specifiedAttList:
            return 0.0
        matched_treatments = 0
        for attribute_index in classifier.specifiedAttList:
            if attribute_index in self.treatment_indices:
                matched_treatments += 1
        if matched_treatments == 0:
            return 0.0
        return min(1.0, matched_treatments / float(len(classifier.specifiedAttList)))