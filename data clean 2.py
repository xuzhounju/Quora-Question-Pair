#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:19:53 2017

@author: joe
"""

import csv
from collections import defaultdict
import math
import gensim
from sklearn import svm
import numpy as np
import time
import re
import copy
from nltk.stem import SnowballStemmer

def readTrainFile():
    f= open('../train.csv','rb')
    reader = csv.reader(f)
    rawDataMatrix=[]
    for row in reader:
        rawDataMatrix.append(row)
    rawDataMatrix=rawDataMatrix[1:]
    
    for pair in rawDataMatrix:
        if len(pair[3])==0 or len(pair[4])==0:
            rawDataMatrix.remove(pair)
        pair[3]= pair[3].split()
        pair[4]= pair[4].split()
    return rawDataMatrix

def dataClean(rawDataMatrix):
    '''
    Convert words to lower case and get rid of punctuation.
    '''
    delims = ',.?!:;'
    for pair in rawDataMatrix:
        pair[3] = map(lambda t: t.lower(), pair[3])
        pair[4]= map(lambda t: t.lower(), pair[4])
        for i in range(len(pair[3])):
            pair[3][i]=pair[3][i].strip(delims)
        for i in range(len(pair[4])):
            pair[4][i]=pair[4][i].strip(delims)
    
    return rawDataMatrix


def dataClean2(rawDataMatrix,stem_word=False):
    for pair in rawDataMatrix:
        pair[3]=" ".join(pair[3])
        pair[4]=" ".join(pair[4])
        pair[3]=cleartext(pair[3],stem_word)
        pair[4]=cleartext(pair[4],stem_word)
        pair[3]=pair[3].split()
        pair[4]=pair[4].split()
    return rawDataMatrix



def cleartext(text,stem_words=False):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    return text
        

def readTestFile():
    f=open('../test.csv','rb')
    reader = csv.reader(f)
    rawDataMatrix=[]
    for row in reader:
        rawDataMatrix.append(row)
    rawDataMatrix=rawDataMatrix[1:]

    return rawDataMatrix

def word2vecTrain(rawDataMatrix,count,dimension):
    print "build up word2vec model..."
    start=time.time()
    sentences= []
    for pair in rawDataMatrix:
        sentences.append(pair[3])
        sentences.append(pair[4])
    model = gensim.models.Word2Vec(sentences, min_count=count,size = dimension)
    end=time.time()
    print "finish the building, time took:",end-start
    return model


def dictAllWords(rawDataMatrix):
    allWords=defaultdict(float)
    for pair in rawDataMatrix:
        for word in pair[3]:
            allWords[word]+=1.0
        for word in pair[4]:
            allWords[word]+=1.0

def wordMatchShare(Q1,Q2,model):
    shareNum= 0.0
    for word in Q1:
        if word in Q2:
            shareNum+=1.0
    return shareNum/(len(Q1)+len(Q2))
        
    

def baselineCosineSimilarity(Q1,Q2,model):
    Q1_dict = defaultdict(float)
    Q2_dict = defaultdict(float)
    for word in Q1:
        Q1_dict[word]+=1.0
    for word in Q2:
        Q2_dict[word]+=1.0
    ab=0.0
    a0=0.0
    b0=0.0
    for token in Q1_dict:
        ab+=Q1_dict[token]*Q2_dict[token]
        a0+=Q1_dict[token]*Q1_dict[token]
    for token in Q2_dict:
        b0+=Q2_dict[token]*Q2_dict[token]
    if b0==0.0 or a0 == 0.0:
        a0=1
        b0=1
    return ab/math.sqrt(a0)/math.sqrt(b0)


def word2vecCosineSimilarity(s1,s2,model):
    Q1,Q2 = word2vecSentence2Vec(s1,s2,model)
    ab = np.dot(Q1,Q2)
    a0 = math.sqrt(np.dot(Q1,Q1))
    b0 = math.sqrt(np.dot(Q2,Q2))
    if b0 == 0.0 or a0 == 0.0:
        a0 = 1.0
        b0 = 1.0
    return ab/a0/b0

def avgPerceptronTrain(example,r,numpass,cosineSimilarity,model):
    print 'start training...'
    t=1.0
    theta=0.5
    s=0.0
    #count = 100000*numpass
    for pair in example:
        #if (t-1)%count==0:
        #    print 'trained ', (t-1)/count,'*100k pairs'
        gold = int(pair[-1])
        for iteration in range(numpass):
            if cosineSimilarity(pair[3],pair[4],model)>theta:
                predict = 1.0
            else:
                predict = 0.0
            g = predict- gold
            theta = theta + r* g
            s = s+(t-1)*r*g
            t+=1
    print 'finished training.learning rate:',r,' ;numpass:',numpass
    return theta -1.0/t*s     

def word2vecSentence2Vec(a,b,model):
    sentenceA = a
    sentenceB = b
    if len(a)!=0 and len(b)!=0:
        vectorA =np.copy(model[sentenceA[0]])
        vectorB =np.copy( model[sentenceB[0]]) 
        for word in sentenceA[1:]:
            vectorA += model[word]
        for word in sentenceB[1:]:
            vectorB += model[word]
        return vectorA/len(a),vectorB/len(b)
    else:
        return [0.0]*len(model['bad']),[1.0]*len(model['bad'])

#def word2vecSVMTrain(rawDataMatrix,model):
#    num=0
#    start = time.time()
#    print "start svm training..."
#    x=[]
#    y=[]
#    for pair in rawDataMatrix:
#        num=num+1
#        x.append(word2vecSentenceDistance(pair[3],pair[4],model))
#        y.append(int(pair[-1]))
#        if num%10000 ==0:
#            print num/10000
#   clf= svm.SVC()
#    clf.fit(x,y)
#    end= time.time()
#    print "finish svm training, time took:",end - start
#    return clf
    



def validation(example,theta,cosineSimilarity,model):
    print 'start validation...'
    num= 0.0 
    total = len(example)
    #count = 0
    for pair in example:
        #count+=1
        gold = int(pair[-1])
        #if count % 100000 == 0:
        #   print 'tested ', count/100000,'*100k pairs'
        if cosineSimilarity(pair[3],pair[4],model)>theta:
            predict = 1.0
        else:
            predict = 0.0
        if gold == predict:
            num+=1.0
    print 'the accuracy is :', num/total
    

    
'''
def word2vecSVMValidation(example,clf,model):
    print "start svm validation..."
    start = time.time()
    num=0.0
    count = 0
    total= len(example)
    for pair in example:
        count+=1
        if count%10000==0:
            print count/10000
        gold = int(pair[-1])
        if clf.predict([word2vecSentenceDistance(pair[3],pair[4],model)])[0]==gold:
            num+=1.0
    end = time.time()
    print "time took:", end - start
    print 'the accuracy of svm is :',num/total
'''


def test(example,theta):
    result=[]
    result.append(['test_id','is_duplicate'])
    for pair in example:
        if baselineCosineSimilarity(pair[1],pair[2])>theta:
            result.append([int(pair[0]),1])
        else:
            result.append([int(pair[0]),0])
    f = open('result.csv',"wb")
    
    writer= csv.writer(f)
    for i in result:
        writer.writerow(i)
    f.close()
    

rawDataMatrix =  readTrainFile()

rawDataMatrix_copy = copy.deepcopy(rawDataMatrix)
rawDataMatrix_copy2 = copy.deepcopy(rawDataMatrix)
rawDataMatrix_copy3 = copy.deepcopy(rawDataMatrix)

model=word2vecTrain(rawDataMatrix,1,200)
cleanedDataMatrix = dataClean(rawDataMatrix_copy2)
model2 = word2vecTrain(cleanedDataMatrix,1,200)
cleanedDataMatrix2 = dataClean2(rawDataMatrix_copy)
model3 = word2vecTrain(cleanedDataMatrix2,1,200)
cleanedDataMatrix3 = dataClean2(rawDataMatrix_copy3,stem_word=True)
model4 = word2vecTrain(cleanedDataMatrix3,1,200)


for r in [0.05]:
    print "******************************************************"
    print "learning rate:",r

        
    print len(rawDataMatrix)
    print "................raw data ....................."
    
    #clf = word2vecSVMTrain(rawDataMatrix,model)
    #word2vecSVMValidation(rawDataMatrix,clf,model)
    
    
    print "dict cosine similarity:"
    threthod1 =avgPerceptronTrain(rawDataMatrix,r,10,baselineCosineSimilarity,model)
    validation(rawDataMatrix,threthod1,baselineCosineSimilarity,model)
    
    print '........................................................'
    print "word2vec cosine similarity:"
    threthod2 = avgPerceptronTrain(rawDataMatrix,r,10,word2vecCosineSimilarity,model)
    validation(rawDataMatrix,threthod2,word2vecCosineSimilarity,model)
    
    print '........................................................'
    print "word match share:"
    threthodA = avgPerceptronTrain(rawDataMatrix,r,10,wordMatchShare,model)
    validation(rawDataMatrix,threthodA,wordMatchShare,model)
    
    
    print ".................cleaned data version 1.............."
    
    print "dict cosine similarity:"
    
    threthod3 = avgPerceptronTrain(cleanedDataMatrix,r,10,baselineCosineSimilarity,model2)
    validation(cleanedDataMatrix,threthod3,baselineCosineSimilarity,model2)
    print '........................................................'

    print "word2vec cosine similarity:"
    
    threthod4 = avgPerceptronTrain(cleanedDataMatrix,r,10,word2vecCosineSimilarity,model2)
    validation(cleanedDataMatrix,threthod4,word2vecCosineSimilarity,model2)
        
    print '........................................................'
    print "word match share:"
    threthodB = avgPerceptronTrain(cleanedDataMatrix,r,10,wordMatchShare,model2)
    validation(cleanedDataMatrix,threthodB,wordMatchShare,model2)
    
    
    print ".................cleaned data version 2.............."
    
    
    print "dict cosine similarity:"
    
    threthod5 = avgPerceptronTrain(cleanedDataMatrix2,r,10,baselineCosineSimilarity,model3)
    validation(cleanedDataMatrix2,threthod5,baselineCosineSimilarity,model3)
    
    print '........................................................'

    print "word2vec cosine similarity:"
    
    threthod6 = avgPerceptronTrain(cleanedDataMatrix2,r,10,word2vecCosineSimilarity,model3)
    validation(cleanedDataMatrix2,threthod6,word2vecCosineSimilarity,model3)

    print '........................................................'
    print "word match share:"
    threthodC = avgPerceptronTrain(cleanedDataMatrix2,r,10,wordMatchShare,model3)
    validation(cleanedDataMatrix2,threthodC,wordMatchShare,model3)
    
    
    
    
    print ".................cleaned data version 3.............."
    
   
    print "dict cosine similarity:"
    
    threthod7 = avgPerceptronTrain(cleanedDataMatrix2,r,10,baselineCosineSimilarity,model3)
    validation(cleanedDataMatrix3,threthod5,baselineCosineSimilarity,model3)
    
    print '........................................................'

    print "word2vec cosine similarity:"
    
    threthod8 = avgPerceptronTrain(cleanedDataMatrix3,r,10,word2vecCosineSimilarity,model4)
    validation(cleanedDataMatrix3,threthod6,word2vecCosineSimilarity,model4)
    
    print '........................................................'
    print "word match share:"
    threthodD = avgPerceptronTrain(cleanedDataMatrix3,r,10,wordMatchShare,model4)
    validation(cleanedDataMatrix3,threthodD,wordMatchShare,model4)
    

#testData = readTestFile()
#result = test(testData,threthod)