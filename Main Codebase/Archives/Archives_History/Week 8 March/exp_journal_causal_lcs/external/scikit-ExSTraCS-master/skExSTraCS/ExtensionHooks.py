import copy


class DefaultFitnessPolicy:
    def calculate(self, model, classifier):
        return pow(classifier.accuracy, model.nu)


class DefaultParentSelectionPolicy:
    def select(self, model, population):
        if model.selection_method == "roulette":
            select_list = population.selectClassifierRW()
        else:
            select_list = population.selectClassifierT(model)
        return select_list[0], select_list[1]


class DefaultSubsumptionPolicy:
    def can_subsume(self, model, parent, child):
        return child.phenotype == parent.phenotype and parent.isSubsumer(model) and self.is_more_general(model, parent, child)

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


def clone_initial_population(initial_population):
    if initial_population is None:
        return None
    return copy.deepcopy(initial_population)