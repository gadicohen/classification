from Classifier import *
import numpy as np
import math

"""
Your NBClassifier dudeNone
"""
class NaiveBayesClassifier(Classifier):

	def learn(self, X, y):
		"""
		You should set up your various counts to be used in classification here: as detailed in the handout.
		Args: 
			X: A list of feature arrays where each feature array corresponds to feature values
				of one observation in the training set.
			y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
				to a negative instance of the class at said 0 or 1s index in featuresList.

		Returns: Nothing
		"""
		 # YOU IMPLEMENT -- CORRECTLY GET THE COUNTS
		"""	#determine:
		 	#number of times class appears in data
		 	# number of obs in data
		 	#number of occurences of each feature xi across all obs labeled as class y
		 	#total number of occurences of features in class  (sum of all feature values across all obs in class)"""

		self.totalFeatureObservations = 0
		self.totalOccurencesOfFeatureInClass=0
		self.totalOccurencesOfFeatureInNotClass=0
		self.occurencesOfClass=0 
		self.occurencesOfNotClass =0
		self.totalFeatureCountsForNotClass = []
		self.totalFeatureCountsForClass=np.zeros(len(X[0]),np.int32)
		self.totalFeatureCountsForNotClass=np.zeros(len(X[0]),np.int32)
		
		featureCountClass=dict()
		featureCountNotClass=dict()


		for x in range(len(X)):
			array=X[x] #array=observation
			if y[x]==1: 
				self.occurencesOfClass+=1 #count total occurrences of class 	
				for feature in range(len(array)): #for every index of feature in the observation
					if feature in featureCountClass.keys(): 
						featureCountClass[feature]+=array[feature] #add to dictionary
					else:
						featureCountClass[feature]=array[feature] #initialize to dictionary
					self.totalOccurencesOfFeatureInClass+=array[feature] 
			if y[x]==0:
				self.occurencesOfNotClass+=1 #count total occurrences of not class
				for feature in range(len(array)):
					if feature in featureCountNotClass.keys():
						featureCountNotClass[feature]+=array[feature] #count feature occurrences in not class
					else:
						featureCountNotClass[feature]=array[feature]
					self.totalOccurencesOfFeatureInNotClass+=array[feature]

		for feature in featureCountClass.values():
				self.totalFeatureObservations+=feature #count total observations
		for feature in featureCountNotClass.values():
				self.totalFeatureObservations+=feature #count total observations


		self.totalFeatureCountsForClass=featureCountClass
		self.totalFeatureCountsForNotClass=featureCountNotClass


	def getLogProbClassAndLogProbNotClass(self, x):

		"""
		You should calculate the log probability of the class/ of not the class using the counts determined
		in learn as detailed in the handout. Don't forget to use epsilon to smooth when a feature in the 
		observation only occurs in only the class or only not the class in the training set! 

		Args: 
			x: a numpy array corresponding to a featurization of a single observation 
			
		Returns: A tuple of (the log probability that the features arg corresponds to a positive 
			instance of the class, and the log probability that the features arg does not correspond
			to a positive instance of the class).
		"""		




		self.epsilon=float(1)/self.totalFeatureObservations

		totalOccurences=(self.occurencesOfClass+self.occurencesOfNotClass)

		PY=float(self.occurencesOfClass) / totalOccurences #get PY
		PNotY=float(self.occurencesOfNotClass) / totalOccurences #get PNotY

		PXgivenY={}
		PXgivenNotY={}
		totalSumPXgivenY=0
		totalSumPXgivenNotY=0

		for feature,featureCount in self.totalFeatureCountsForClass.iteritems():
			if featureCount==0 and self.totalFeatureCountsForNotClass[feature]!=0:
				self.totalFeatureCountsForClass[feature]=self.epsilon
		for feature,featureCount in self.totalFeatureCountsForNotClass.iteritems():
			if featureCount==0 and self.totalFeatureCountsForClass[feature]!=0:
				self.totalFeatureCountsForNotClass[feature]=self.epsilon

		for feature,featureCount in self.totalFeatureCountsForClass.iteritems():
			PXgivenY[feature]=float(featureCount)/self.totalOccurencesOfFeatureInClass
		for feature,featureCount in self.totalFeatureCountsForNotClass.iteritems():
			PXgivenNotY[feature]=float(featureCount)/self.totalOccurencesOfFeatureInNotClass


		totalSumPXgivenY = sum([x[i]*math.log(PXgivenY[i]) for i in range(len(PXgivenY)) if PXgivenY[i]!=0])
		totalSumPXgivenNotY = sum([x[i]*math.log(PXgivenNotY[i]) for i in range(len(PXgivenNotY)) if PXgivenNotY[i]!=0])
		
		logProbClass = math.log(PY)+totalSumPXgivenY
		logProbNotClass = math.log(PNotY)+totalSumPXgivenNotY

		return (logProbClass, logProbNotClass)







