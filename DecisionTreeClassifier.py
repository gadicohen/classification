from Classifier import *
import math
import random

class DecisionTreeClassifier(Classifier): 

    def learn(self, X, y):
        """
        Constructs a decision tree.

        Args:
           X: A list of feature arrays where each feature array corresponds to feature values
        of one observation in the training set.
           y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
        to a negative instance of the class at said 0 or 1s index in featuresList.
        """
        DT = TreeNode(X, y, 0)
        DT.makeTree()
        self.DT = DT

    def getLogProbClassAndLogProbNotClass(self, x):
        """Returns log probabilities that a given observation is a positive sample or negative sample"""
        return self.DT.getLogProbClassAndLogProbNotClass(x)

class TreeNode: 

    def __init__(self, X, y, depth):
        self.X = X  # set of featurized observations
        self.y = y  # set of labels associated with the observations 
        self.depth = depth
        self.depthLimit = 10  # limits the depth of your tree for the sake of performance; feel free to adjust
        self.n = len(X)
        self.splitFeature, self.children = None, None  # these attributes should be assigned in splitNode()
        self.entropySplitThreshold = 0.7219 # node splitting threshold for 80%/20% split; feel free to adjust

    def splitNode(self, splitFeature):
        ''' Creates child nodes, splitting the featurized data in the current node on splitFeature. 
        Must set self.splitFeature and self.children to the appropriate values.

        Args: splitFeature, the feature on which this node should split on (this should be the feature you obtain from
            the bestFeature() function)
        Returns: returns True if split is performed, False if not.
        '''
        if len(set(self.y)) < 2: # fewer than 2 labels in this node, so no split is performed (node is a leaf)
            return False
        
        if splitFeature==None:
            return False

        self.splitFeature=splitFeature

        trueList=[]
        falseList=[]
        trueLabels=[]
        falseLabels=[]

        y=self.y
        
        for x in range(len(self.X)):
            array=self.X[x]
            if array[splitFeature] > 0:
                trueList.append(array)
                trueLabels.append(y[x])
            if array[splitFeature] == 0:
                falseList.append(array)
                falseLabels.append(y[x])

        trueChild=TreeNode(trueList,trueLabels,self.depth+1)
        falseChild=TreeNode(falseList,falseLabels,self.depth+1)
        self.children=trueChild,falseChild

        return True

    def bestFeature(self):
        ''' Identifies and returns the feature that maximizes the information gain.
        You should calculate entropy values for each feature, and then return the feature with highest entropy.
        Consider thresholding on an entropy value -- that is, select a target entropy value, and if no feature 
        has entropy above that value, return None as the bestFeature 

        Returns: the index of the best feature based on entropy
        '''

        self.entropies={}
        
        maxEnt=-float("inf")
        maxKey=-1

        positives={k:0 for k in range(len(self.X[0]))}
        negatives={k:0 for k in range(len(self.X[0]))}
        totalFeatures=0

        for observation in range(len(self.X)):
            array=self.X[observation]
            for feature in range(len(array)):
                if array[feature] == 0:
                    negatives[feature]+=1
                elif array[feature]!=0:
                    positives[feature]+=1

        for feature in range(len(self.X[0])):
            totalFeature=len(self.X)
            probPositive=float(positives[feature])/totalFeature
            probNegative=float(negatives[feature])/totalFeature
            #print probPositive, probNegative,totalFeature
            
            if probPositive == float(0):
                probPositive=float(1)/float(1000)
            if probNegative == float(0):
                probNegative=float(1)/float(1000)

            self.entropies[feature]=probPositive*-math.log(probPositive,2)+probNegative*-math.log(probNegative,2)

            if self.entropies[feature]>=maxEnt:
                maxEnt=self.entropies[feature]
                maxKey=feature

        bestFeature = maxKey

        if self.entropies[maxKey]<self.entropySplitThreshold:
            bestFeature=None
        
        return bestFeature



    def makeTree(self):
        '''Splits the root node on the best feature (if applicable),
        then recursively calls makeTree() on the children of the root.
        If there is no best feature, you should not perform a split, and this
        node will become a leaf'''

        if self.depth <= self.depthLimit:
            bestFeature = self.bestFeature()
            if bestFeature != None:
                split=self.splitNode(bestFeature)
                if self.children!=None:
                    for child in self.children:
                        child.makeTree()

    def getLogProbClassAndLogProbNotClass(self, x):
        """
        Args:
            x: A numpy array that corresponds to the feature values for a single observation.

        Returns:
            A tuple containing the log probability that the observation is a member of the class
                and the log probability that the observation is NOT a member of the class
        """


        while self.children!=None:
            if x[self.splitFeature]>0:
                self=self.children[0]
            else:
                self=self.children[1]

        positive=0
        negative=0

        for observation in range(len(self.X)):
            if self.y[observation]==0:
                negative+=1
            if self.y[observation]==1:
                positive+=1
        total=len(self.X)

        positives={k:0 for k in range(len(x))}
        negatives={k:0 for k in range(len(x))}
        totalFeatures=0

        print positive, negative, total
        if positive==0:
            probClass=math.log(float(1)/1000)
        else:
            probClass=math.log(float(positive)/total)

        if negative==0:
            probNotClass=math.log(float(1)/1000)
        else:
            probNotClass=math.log(float(negative)/total)

        return (probClass, probNotClass)
