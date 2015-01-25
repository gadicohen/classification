import math

class Classifier:
	def __init__(self, featsAndLabels):
		features, labels = featsAndLabels
		self.learn(features, labels)

	def getLogProbClassAndLogProbNotClass(self, features):
		pass

	def classify(self, features):
		logProbClass, logProbNotClass = self.getLogProbClassAndLogProbNotClass(features)
		if logProbClass > logProbNotClass:
			return 1
		return 0

	def learn(self, featuresList, labels):
		pass

	def getConfidence(self, listOfFeatures):
		toReturn = []
		for features in listOfFeatures:
			logProbClass, logProbNotClass = self.getLogProbClassAndLogProbNotClass(features)
			confidence = logProbClass - logProbNotClass
			toReturn.append(confidence)
		return toReturn
