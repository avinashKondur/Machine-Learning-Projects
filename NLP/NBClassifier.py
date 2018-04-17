import numpy as np
from io import StringIO
import re
from collections import Counter
from decimal import Decimal
import mlutils as ml

class NBClassifier:
	
    def getUniGrams(data):
        words=[]
        for line in data:
            line = re.sub(r"[,.'-;:|\n]", "", line)
            line = line.strip().lower()
            w = line.split(' ')
            words.extend(w)
        return words

    def getBiGrams(data):
        biGrams = []
        for line in data:
            line = re.sub(r"[,.'-;:|\n]", "", line)
            line = line.strip().lower()
            w = line.split(' ')
            for i in range(len(w)):
                if i ==0 :
                    biGrams.append('_'+','+w[i])
                elif i == len(w)-1:
                    biGrams.append(w[i]+','+'_')
                else:
                    biGrams.append(w[i]+','+w[i+1])
        return biGrams
    
    def train(XNtrain, XPtrain):
        #get negative unigrams and Bigrams
        nWords = getUniGrams(XNtrain)
        nWords.extend(getBiGrams(XNtrain))

        #get Posititve uniGrams and BiGrams
        pWords = getUniGrams(XPtrain)
        pWords.extend(getBiGrams(XPtrain))

        #get counts for each words
        negWordCounts = Counter(nWords)
        posWordCounts = Counter(pWords)

        totalNegWords = len(nWords)
        totalPosWords = len(pWords)

        print('totalNegWords = ',totalNegWords)
        print('totalPosWords = ',totalPosWords)

        totalNumberOfWords = totalNegWords + totalPosWords

        probNegClass = Decimal(totalNegWords/totalNumberOfWords)
        probNegClass = round(probNegClass,8)
        probPosClass = Decimal(totalPosWords/totalNumberOfWords)
        probPosClass = round(probPosClass,8)

        negWordProbs = dict()
        for k in negWordCounts.keys():
            count = negWordCounts[k]
            prob = Decimal(count/totalNegWords)
            prob = round(prob,8)
            negWordProbs[k] = prob

        posWordProbs = dict()
        for k in posWordCounts.keys():
            count = posWordCounts[k]
            prob = Decimal(count/totalPosWords)
            prob = round(prob,8)
            posWordProbs[k] = prob

        return {'probNegClass':probNegClass, 'probPosClass':probPosClass,
                'negWordProbs':negWordProbs,'posWordProbs':posWordProbs, 
                'totalNegWords':totalNegWords, 'totalPosWords':totalPosWords}
    def use(XTest, Ttest, model):
   
        predicted = np.zeros((1,1),dtype=np.int)
        for i in range(len(XTest)):
            line = re.sub(r"[,.'-;:|\n]", "", XTest[i])
            line = line.strip().lower()
            w=[]
            w.append(line)
            nWords = getUniGrams(w)
            nWords.extend(getBiGrams(w))
            lineSenti = getSentiment(nWords,model)
            if i==0:
                predicted[0] = lineSenti 
            else:
                predicted = np.concatenate((predicted,np.array([[lineSenti]],dtype=np.int)),axis=0)


        return (Decimal(np.sum(Ttest==predicted)/len(XTest)) * 100)
    
    def getSentiment(words, model):
        probNegClass = model['probNegClass']
        probPosClass = model['probPosClass']

        negProb = 1
        posProb = 1

        negWordsCount = model['totalNegWords']
        posWordsCount = model['totalPosWords']

        negSet = model['negWordProbs']
        posSet = model['posWordProbs']        
        negKeySet = negSet.keys()
        posKeySet = posSet.keys()

        for w in words:
            if w in negKeySet:
                negProb*=negSet[w]
            else:
                prob = round(Decimal(1/negWordsCount),8)
                negProb*=prob

            if w in posKeySet:
                posProb*=posSet[w]
            else:
                prob = round(Decimal(1/posWordsCount),8)
                posProb*=prob

        negProb*=probNegClass
        posProb*=probPosClass

        return 0 if negProb > posProb else 1
