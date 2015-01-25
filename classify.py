import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from FeatureGetters import *

from NaiveBayesClassifier import *
from LogisticRegressionClassifier import *
from DecisionTreeClassifier import *
import FeatureGetters

def runNBClassifier(train, test):
    print "building naive bayes classifier"
    model = NaiveBayesClassifier(train)
    validate(model, "naive bayes", test)
    return model

def runLRClassifier(train, test):
    print "building logistic regression classifier"
    model = LogisticRegressionClassifier(train)
    validate(model, "logistic regression", test)
    return model

def runDecisionTreeClassifier(train, test):
    print "building decision tree classifier"
    model = DecisionTreeClassifier(train)
    validate(model, "decision tree", test)

def validate(model, modelname, test):
    featuresList, labels = test
    n = len(featuresList) # n = number of observations
    print "classifying test data for "+ modelname +" model"
    classification_accuracy = sum( 1 for i in range(n) if labels[i] == model.classify(featuresList[i]) ) / float(n)
    print "Classification test accuracy: ", classification_accuracy

def plotRoc(test, model):

    print "Plotting ROCNone"

    confidences = model.getConfidence(test[0])

    fpr, tpr, threshold = roc_curve(test[1], confidences) 
    roc_auc = auc(fpr,tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    pl.clf()
    pl.plot(fpr,tpr,label='ROC curve (area = %-.2f)' % roc_auc)
    pl.plot([0,1], [0,1],'k--')
    pl.xlim([0.0,1.0])
    pl.ylim([0.0,1.0])
    pl.xlabel("False Positive Rate")
    pl.ylabel("True Positive Rate")
    pl.title("Receiver operating characteristic")
    pl.legend(loc="lower right")
    pl.show()


def getModel(trainData, modelName):
    if modelName == "naive":
        return NaiveBayesClassifier(trainData)
    elif modelName == "logreg":
        return LogisticRegressionClassifier(trainData)
    elif modelName == "tree":
        return DecisionTreeClassifier(trainData)
    else:
        printUsage()



def printUsage():
    print "Usage: python classify.py 1st_arg [2nd_arg]\n\t" + \
            "First argument options:\n\t\t" + \
            "<naive>: your naive bayes classifier\n\t\t" + \
            "<logreg>: your logistic regression classifier\n\t\t" + \
            "<tree>: your decision tree classifier\n\t\t" + \
            "<roc>: your ROC curve and accompanying metrics\n\t" + \
            "2nd argument:\n\t\t" + \
            "<door> : runs the classifier on the door domain\n\t\t" + \
            "<email> : runs the classifier on the email domain" + \
            "3rd argument(optional):\n\t\t" + \
            "<roc> : plot the roc curve instead of testing" + \
    quit()

if __name__ == "__main__":

    import sys
    argc = len(sys.argv)

    if not argc in (3, 4):
        printUsage()

    if sys.argv[2] == "door":
        print "Featurizing door dataNone"
        trainData = DoorFeatures("./data/doors/train").getFeatures()
        testData = DoorFeatures("./data/doors/test").getFeatures()


    elif sys.argv[2] == "email":
        emailTrainFileLoc = "./data/train_small.txt"
        emailTestFileLoc = "./data/test_small.txt"
        print "Featurizing email dataNone"

        # Tune featurization for different classifiers for emails
        if sys.argv[1] == "naive":
            maxWords = 5000
            downSamplingRate = 1

        elif sys.argv[1] == "logreg":
            maxWords = 100
            downSamplingRate = 1

        elif sys.argv[1] == "tree":
            emailTrainFileLoc = "./data/train.txt"
            emailTestFileLoc = "./data/test.txt"
            maxWords = 100 
            downSamplingRate = 1
        else:
            printUsage()

        emailFeatureGetter = EmailFeatureGetter(maxWords, downSamplingRate, emailTrainFileLoc)
        trainData = emailFeatureGetter.getFeaturesAndLabelsFromFileLoc(emailTrainFileLoc)
        testData = emailFeatureGetter.getFeaturesAndLabelsFromFileLoc(emailTestFileLoc)

    else:
        printUsage()

    if argc == 4 and sys.argv[3] == "roc":
        model = getModel(trainData, sys.argv[1])
        plotRoc(testData, model)
        quit()
    if argc == 4 and not sys.argv[3] == "roc":
        printUsage()

    if sys.argv[1] == "naive":
        runNBClassifier(trainData,testData)

    elif sys.argv[1] == "logreg":
        runLRClassifier(trainData, testData)

    elif sys.argv[1] == "tree":
        runDecisionTreeClassifier(trainData, testData)

    else:
        printUsage()

