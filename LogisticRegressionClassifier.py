from Classifier import *
import numpy as np
import math

class LogisticRegressionClassifier(Classifier):

    def learn(self, X, y):
        """
        Performs an iterative gradient ascent on the weight values.
        
        Args:
	       X: A list of feature arrays where each feature array corresponds to feature values
		of one observation in the training set.
	       y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
		to a negative instance of the class at said 0 or 1s index in featuresList.

	Returns: Nothing
        """
         # learning rate
        self.eta = .001
        # convergence threshold
        self.epsilon = 0.01


        #x is the list of features of o
        #each feature value in x has a pre-determined w
        #the meat of logistic regression is to find the weights vector w that maximizes the probability that each observation in the training set is correctly classified
        #w is found iteratively through gradient descent
        
        weights={}

        #initialize weights
        j=0
        for feature in X[1]:
            weights[j]=.0001
            j+=1

        averageWChange=.0001
        lastAverageWChange=0


        errors={}
        difference=math.fabs(float(averageWChange-lastAverageWChange)/averageWChange)

        while difference>=self.epsilon:


            allWeightChanges=[]
            w0=1

            #get all the error observations
            for observation in range(len(X)):
                yValue=y[observation]
                weightfeatureSum=0
                for feature in range(len(X[observation])):
                    weightfeatureSum+=(X[observation][feature]*weights[feature])
                Pofoisy=float(math.exp(w0+weightfeatureSum))/(1+math.exp(w0+weightfeatureSum))
                errors[observation]=(yValue-Pofoisy)

            weightChanges={i:0 for i in range(len(X[0]))}

            #find all weight changes
            for observation in range(len(X)):
                for feature in range(len(X[observation])):
                    weightChanges[feature]+=X[observation][feature]*errors[observation]
                    print feature,weightChanges[feature],X[observation][feature],errors[observation]


            #set new weights
            for observation in range(len(X)):
                array=X[observation]
                for feature in range(len(array)):
                    weights[feature]+=self.eta*weightChanges[feature]
                    allWeightChanges.append(weightChanges[feature])
                    averageWChange=sum(allWeightChanges)/len(allWeightChanges)

            difference=math.fabs(float(averageWChange-lastAverageWChange)/lastAverageWChange)
            lastAverageWChange=averageWChange

        self.weights=weights

    def getLogProbClassAndLogProbNotClass(self, x):
        """
        Args:
            features: A numpy array that corresponds to the feature values for a single observation.

        Returns:
            A tuple containing the log probability that the observation is a member of the class
                and the log probability that the observation is NOT a member of the class
        """

        w0=1

        weightfeatureSum=0
        for feature in range(len(x)):
            weightfeatureSum+=(x[feature]*self.weights[feature])
            print feature,weightfeatureSum, x[feature],self.weights[feature]
        Pofoisy=float(math.exp(w0+weightfeatureSum))/(1+math.exp(w0+weightfeatureSum))

        if Pofoisy!=0:
            logProbClass = math.log(Pofoisy)
            logProbNotClass = math.log(1-Pofoisy)
        else:
            logProbClass=0
            logProbNotClass=1

        return (logProbClass, logProbNotClass)
