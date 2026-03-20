class CausalSubsumptionPolicy:
    def __init__(self, metadata):
        self.metadata = metadata

    def can_subsume(self, model, parent, child):
        if child.phenotype != parent.phenotype:
            return False
        if not parent.isSubsumer(model):
            return False
        if not self.is_more_general(model, parent, child):
            return False
        return self._get_parent_nodes(parent) == self._get_parent_nodes(child)

    def is_more_general(self, model, candidate, other):
        if len(candidate.specifiedAttList) >= len(other.specifiedAttList):
            return False
        for i in range(len(candidate.specifiedAttList)):
            attribute_index = candidate.specifiedAttList[i]
            attribute_info_type = model.env.formatData.attributeInfoType[attribute_index]
            if attribute_index not in other.specifiedAttList:
                return False

            if attribute_info_type:
                other_ref = other.specifiedAttList.index(attribute_index)
                if candidate.condition[i][0] > other.condition[other_ref][0]:
                    return False
                if candidate.condition[i][1] < other.condition[other_ref][1]:
                    return False
        return True

    def _get_parent_nodes(self, classifier):
        parent_nodes = set()
        for attribute_index in classifier.specifiedAttList:
            for node in self.metadata.rule_parent_map.get(int(attribute_index), []):
                parent_nodes.add(node)
        return tuple(sorted(parent_nodes))