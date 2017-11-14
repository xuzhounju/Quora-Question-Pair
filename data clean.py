#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:35:26 2017

@author: joe
"""
from collections import defaultdict
import math
f=open('../train.csv')
first_line = f.readline()
print first_line.split(',')

wrong= 0
rawDataMatrix = []
raw = f.read().replace('"\r\n','",')
raw =raw[1:-2].split('","')
print len(raw)
pairId = 1
pair=raw[0:6]
for i in raw[6:]:
    try:
        a=int(i)
        if a==pairId:
            rawDataMatrix.append(pair)
            pair = []
            
            pairId +=1
            pair.append(i)
        else:
            pair.append(i)
    except ValueError:
        pair.append(i)
rawDataMatrix.append(pair)



for pair in rawDataMatrix:
    if len(pair)!=6 or len(pair[3])==0 or len(pair[4])==0:
        rawDataMatrix.remove(pair)
        
        

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
        print Q1
        print Q2
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

def test(example,theta):
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
threthod =avgPerceptronTrain(rawDataMatrix,0.2,10)
test(rawDataMatrix,threthod)