import numpy as np
import random

class Prediction():
    def __init__(self,model,population, state=None):
        self.decision = None
        self.probabilities = {}
        self.hasMatch = len(population.matchSet) != 0

        #Discrete Phenotypes
        if model.env.formatData.discretePhenotype:
            self.vote = {}
            self.tieBreak_Numerosity = {}
            self.tieBreak_TimeStamp = {}

            for eachClass in model.env.formatData.phenotypeList:
                self.vote[eachClass] = 0.0
                self.tieBreak_Numerosity[eachClass] = 0.0
                self.tieBreak_TimeStamp[eachClass] = 0.0

            # MN-LCS Core: Micro-NN voting mechanism
            raw_neural_votes = []
            
            for ref in population.matchSet:
                cl = population.popSet[ref]
                base_vote = cl.fitness * cl.numerosity * model.env.formatData.classPredictionWeights[cl.phenotype]
                
                # Check if micro_mlp has learned any neural weights yet
                if state is not None and hasattr(cl.micro_mlp, 'coefs_') and len(cl.micro_mlp.classes_) > 1:
                    try:
                        mlp_prob = cl.micro_mlp.predict_proba([state])[0]
                        predicted_class_index = np.argmax(mlp_prob)
                        predicted_phenotype = cl.micro_mlp.classes_[predicted_class_index]
                        
                        # Weight the vote by both evolutionary fitness and neural confidence
                        neural_confidence = mlp_prob[predicted_class_index]
                        self.vote[predicted_phenotype] += base_vote * (0.5 + neural_confidence)
                        raw_neural_votes.append((cl, mlp_prob))
                    except Exception: # Fallback to standard symbolic voting if NN fails
                        self.vote[cl.phenotype] += base_vote
                else: # Fallback to standard symbolic voting
                    self.vote[cl.phenotype] += base_vote
                
                self.tieBreak_Numerosity[cl.phenotype] += cl.numerosity
                self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp
                
            # Negative Correlation Learning (NCL) penalty evaluation
            if len(raw_neural_votes) > 1:
                avg_prob = np.mean([v[1] for v in raw_neural_votes], axis=0)
                for cl, m_prob in raw_neural_votes:
                    # Cosine similarity between rule's neural output and ensemble average
                    similarity = np.dot(m_prob, avg_prob) / (np.linalg.norm(m_prob) * np.linalg.norm(avg_prob) + 1e-9)
                    cl.ncl_penalty = similarity # Stored for ExSTraCS fitness deduction to enforce diversity

            #Populate Probabilities
            sProb = 0
            for k,v in sorted(self.vote.items()):
                self.probabilities[k] = v
                sProb += v
            if sProb == 0: #In the case the match set doesn't exist
                for k, v in sorted(self.probabilities.items()):
                    self.probabilities[k] = 0
            else:
                for k,v in sorted(self.probabilities.items()):
                    self.probabilities[k] = v/sProb

            highVal = 0.0
            bestClass = []
            for thisClass in model.env.formatData.phenotypeList:
                if self.vote[thisClass] >= highVal:
                    highVal = self.vote[thisClass]

            for thisClass in model.env.formatData.phenotypeList:
                if self.vote[thisClass] == highVal:  # Tie for best class
                    bestClass.append(thisClass)

            if highVal == 0.0:
                self.decision = None

            elif len(bestClass) > 1:
                bestNum = 0
                newBestClass = []
                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] >= bestNum:
                        bestNum = self.tieBreak_Numerosity[thisClass]

                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] == bestNum:
                        newBestClass.append(thisClass)

                if len(newBestClass) > 1:
                    bestStamp = 0
                    newestBestClass = []
                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                            bestStamp = self.tieBreak_TimeStamp[thisClass]

                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                            newestBestClass.append(thisClass)
                    # -----------------------------------------------------------------------
                    if len(newestBestClass) > 1:  # Prediction is completely tied - extracs has no useful information for making a prediction
                        self.decision = 'Tie'
                else:
                    self.decision = newBestClass[0]
            else:
                self.decision = bestClass[0]

        if self.decision == None or self.decision == 'Tie':
            if model.env.formatData.discretePhenotype:
                self.decision = model.env.formatData.majorityClass
                #self.decision = random.choice(model.env.formatData.phenotypeList)
            else:
                self.decision = random.randrange(model.env.formatData.phenotypeList[0],model.env.formatData.phenotypeList[1],(model.env.formatData.phenotypeList[1]-model.env.formatData.phenotypeList[0])/float(1000))


    def getFitnessSum(self, population, low, high):
        """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
        fitSum = 0
        for ref in population.matchSet:
            cl = population.popSet[ref][0]
            if cl.phenotype[0] <= low and cl.phenotype[1] >= high:  # if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum

    def getDecision(self):
        """ Returns prediction decision. """
        return self.decision

    def getProbabilities(self):
        ''' Returns probabilities of each phenotype from the decision'''
        a = np.empty(len(sorted(self.probabilities.items())))
        counter = 0
        for k, v in sorted(self.probabilities.items()):
            a[counter] = v
            counter += 1
        return a
