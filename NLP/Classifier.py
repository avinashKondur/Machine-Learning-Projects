

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize
import copy

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time


class Classifier:
    
    def __init__(self, classifierName, posFile, negFile):
        
        self._name = classifierName
        
        pos = pd.read_table(posFile,delimiter='\n',header=None, names=["text"] )
        pos['sentiment'] = 1 #1 for positive

        neg = pd.read_table(negFile,delimiter='\n',header=None, names=["text"] )
        neg['sentiment'] = 2 #2 for negative
        
        pos_words=[]
        for s in pos['text']:
           short_p_words.extend(word_tokenize(str(s)))

        neg_words=[]
        for s in neg['text']:
            neg_words.extend(word_tokenize(str(s)))

        all_words=[]
        for w in pos_words:
            all_words.append(w.lower())

        for w in neg_words:
            all_words.append(w.lower())

            
        all_words = nltk.FreqDist(all_words)

        self.word_features = list(all_words.keys())[:int(len(all_words)*0.8)]
        
        documents = pos.get_values()
        documents = np.concatenate((documents,neg.get_values()),axis=0)

        #shuffle the documents
        random.shuffler(documents)

        #prepare X and T, classification
        self.X = document[:,0:1]
        self.T = documents[:,1:2]
        
        if classifierName == 'NaiveBayesClassifier':
            self.classifier = nltk.NaiveBayesClassifier
        elif classifierName == 'MaxEntropy':
            classifier = nltk.MaxentClassifier
        elif classifierName == 'MultinomialNB':
            self.classifier = SklearnClassifier(MultinomialNB())
        elif classifierName == 'BernoulliNB':
            self.classifier = SklearnClassifier(BernoulliNB())
        elif classifierName == 'LogisticRegression':
            self.classifier = SklearnClassifier(LogisticRegression())
        elif classifierName == 'SGDClassifier':
            self.classifier = SklearnClassifier(SGDClassifier())
        elif classifierName == 'LinearSVC':
            self.classifier = SklearnClassifier(SGDClassifier())
        elif classifierName == 'NuSVC':
            self.classifier = SklearnClassifier(SGDClassifier())
        else:
            raise ValueError('Not a valid classifier name')
        
    
    def find_features(self,document):
        words = word_tokenize(document)
        features = {}
        for w in self.word_features:
            features[w] = (w in words)

        return features
    
    def train(self,Xtrain,numIterations = 100, algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]):
        
        print('Training the dataset')
        
        featuresets = [(self.find_features(rev), category) for (rev, category) in Xtrain]
        
        if self._name = 'NaiveBayesClassifier':
            self.classifer.train(featuresets)
            self.classifer.show_most_informative_features(15)
        elif self._name = 'MaxEntropy':
            classifier = nltk.MaxentClassifier.train(featuresets, algorithm, max_iter=numIterations)
            classifier.show_most_informative_features(10)
        else:
            self.classifer.train(featuresets)
        
        print('Training the dataset is done')
    
    def evaluate(self,X,T):
        
        testing_set = [(self.find_features(rev), category) for (rev, category) in X]
        
        results = np.array([[1]])
        i=0
        for review in testing_set:
            label = review[1]
            text = review[0]
            determined_label = classifier.classify(text)
            if i ==0:
                results [0] = 1 if determined_label=='pos' else 2
                i= i+1
            else:
                results = np.concatenate((results,np.array([[1 if determined_label=='pos' else 2]])),axis=0)
        
        
        print("Original {0} accuracy percent:{1}".format(self._name, (nltk.classify.accuracy(self.classifier, testing_set))*100))
        
        #plot the results
        self.Plot(results,T)
        
        return (nltk.classify.accuracy(self.classifier, testing_set))*100, results
    
    def Plot(self,results, T):
        
        style.use("ggplot")

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        
        #start plotting expected results
        xar = []
        yar = []

        x = 0
        y = 0

        for l in T:
            x += 1
            if l==1:
                y += 1
            elif l==2:
                y -= 1

            xar.append(x)
            yar.append(y)
 
        ax1.plot(xar,yar)
        
        #start plotting determined results
        xar = []
        yar = []

        x = 0
        y = 0

        for l in results:
            x += 1
            if l==1:
                y += 1
            elif l==2:
                y -= 1

            xar.append(x)
            yar.append(y)
            
        ax1.plot(xar,yar)
        
        plt.show()
    
    def trainValidateTestKFoldsClassification(self,parameterSets,nFolds,shuffle=False,verbose=False):
        if nFolds < 3:
            raise ValueError('ERROR: trainValidateTestKFoldsClassification requires nFolds >= 3.')
        # Collect row indices for each class
        classes = np.unique(self.T)
        K = len(classes)
        rowIndicesByClass = []
        for c in classes:
            rowsThisClass = np.where(self.T == c)[0]
            if shuffle:
                np.random.shuffle(rowsThisClass)
            rowIndicesByClass.append(rowsThisClass)
        # Collect start and stop indices for the folds, within each class
        startsStops = []
        if verbose:
            print('  In each of',nFolds,'folds, Class-Counts ',"")
        for k,rowIndicesThisClass in enumerate(rowIndicesByClass):
            nSamples = len(rowIndicesThisClass)
            nEach = int(nSamples / nFolds)
            if verbose:
                print('{}-{},'.format(classes[k],nEach), " ") #'samples in each of',nFolds,'folds.')
            if nEach == 0:
                raise ValueError("trainValidateTestKFoldsClassification: Number of samples in each fold for class {} is 0.".format(k))
            startsThisClass = np.arange(0,nEach*nFolds,nEach)
            if k < K-1: #last class
                stopsThisClass = startsThisClass + nEach
            else:
                stopsThisClass = startsThisClass + nSamples #Each
            startsStops.append(list(zip(startsThisClass,stopsThisClass)))
        print()

        results = []
        for testFold in range(nFolds):
            # Leaving the testFold out, for each validate fold, train on remaining
            # folds and evaluate on validate fold. 
            bestParms = None
            bestValidationAccuracy = 0
            for parms in parameterSets:
                validateAccuracySum = 0
                for validateFold in range(nFolds):
                    if testFold == validateFold:
                        continue
                    trainFolds = np.setdiff1d(range(nFolds), [testFold,validateFold])
                    rows = []
                    for tf in trainFolds:
                        for k in range(K):
                            a,b = startsStops[k][tf]                
                            rows += rowIndicesByClass[k][a:b].tolist()
                    Xtrain = self.X[rows,:]
                    Ttrain = self.T[rows,:]
                    # Construct Xvalidate and Tvalidate
                    rows = []
                    for k in range(K):
                        a,b = startsStops[k][validateFold]
                        rows += rowIndicesByClass[k][a:b].tolist()
                    Xvalidate = self.X[rows,:]
                    Tvalidate = self.T[rows,:]

                    self.train(Xtrain)
                    
                    validateAccuracy = self.evaluate(Xvalidate,Tvalidate)
                    
                    validateAccuracySum += validateAccuracy
                    
                validateAccuracy = validateAccuracySum / (nFolds-1)
                
                if bestParms is None or validateAccuracy > bestValidationAccuracy:
                    bestParms = parms
                    bestValidationAccuracy = validateAccuracy
            rows = []
            for k in range(K):
                a,b = startsStops[k][testFold]
                rows += rowIndicesByClass[k][a:b].tolist()
            Xtest = self.X[rows,:]
            Ttest = self.T[rows,:]

            newXtrain = np.vstack((Xtrain,Xvalidate))
            newTtrain = np.vstack((Ttrain,Tvalidate))
            
            self.train(newXtrain,newTtrain)
            
            trainAccuracy = self.evaluate(newXtrain,newTtrain)
            testAccuracy= self.evaluate(Xtest,Ttest)

            resultThisTestFold = [bestParms, trainAccuracy,
                                  bestValidationAccuracy, testAccuracy]
            results.append(resultThisTestFold)
            if verbose:
                print(resultThisTestFold)
        return results  

    def printResults(self,results):
        print('{:4s} {:>20s}{:>8s}{:>8s}{:>8s}'.format('Algo','Parameters','TrnAcc','ValAcc','TesAcc'))
        print('-------------------------------------------------')
        for row in results:
            # 20 is expected maximum number of characters in printed parameter value list
            print('{:>4s} {:>20s} {:7.2f} {:7.2f} {:7.2f}'.format(self._name,str(row[0]),*row[1:])) 
    