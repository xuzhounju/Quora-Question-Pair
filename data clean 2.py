#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:19:53 2017

@author: joe
"""

import csv
from collections import defaultdict
import math


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
    return rawDataMatrix


def readTestFile():
    f=open('../test.csv','rb')
    reader = csv.reader(f)
    rawDataMatrix=[]
    for row in reader:
        rawDataMatrix.append(row)
    rawDataMatrix=rawDataMatrix[1:]

    return rawDataMatrix
    

def dictAllWords(rawDataMatrix):
    allWords=defaultdict(float)
    for pair in rawDataMatrix:
        for word in pair[3].split():
            allWords[word]+=1.0
        for word in pair[4].split():
            allWords[word]+=1.0
    

def baselineCosineSimilarity(Q1,Q2):
    Q1_dict = defaultdict(float)
    Q2_dict = defaultdict(float)
    for word in Q1.split():
        Q1_dict[word]+=1.0
    for word in Q2.split():
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

def avgPerceptronTrain(example,r,numpass):
    t=1.0
    theta=0.5
    s=0.0
    for pair in example:
        gold = int(pair[-1])
        for iteration in range(numpass):
            if baselineCosineSimilarity(pair[3],pair[4])>theta:
                predict = 1.0
            else:
                predict = 0.0
            g = predict- gold
            theta = theta + r* g
            s = s+(t-1)*r*g
            t+=1
        
    return theta -1.0/t*s                

def validation(example,theta):
    num= 0.0 
    total = len(example)
    for pair in example:
        gold = int(pair[-1])

        if baselineCosineSimilarity(pair[3],pair[4])>theta:
            predict = 1.0
        else:
            predict = 0.0
        if gold == predict:
            num+=1.0
    print 'the accuracy is :', num/total

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
threthod =avgPerceptronTrain(rawDataMatrix,0.2,10)
validation(rawDataMatrix,threthod)
testData = readTestFile()
result = test(testData,threthod)