# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:32:07 2018

@author: Dhruv
"""

from random import *
import math
import time
import operator


import numpy as np
from sklearn import svm , metrics ,datasets, neighbors, linear_model
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.spatial import distance as dist
import copy

MAX_ITER = 50
POP_SIZE = 49
MAX_FEATURE = 19
V_MAX = 6
V_MIN = 0

trainX = []
trainY = []
testX =  []
testY =  []

Population = []
Universes = []
SortedUniverses = []
WEP_Max=1;
WEP_Min=0.2;
BestUniverse = []
BestCost = 0


def readData():
	global trainX, trainY, testX, testY
	f = open('gi/train_inp.txt','rb')
	for line in f:
		trainX.append([float(i) for i in line.split()])
	f = open('gi/train_out.txt','rb')
	for i in f:
		trainY.append(int(i))
	f = open('gi/test_inp.txt','rb')
	for line in f:
		testX.append([float(i) for i in line.split()])
	f = open('gi/test_out.txt','rb')
	for i in f:
		testY.append(int(i))
	trainX = np.array(trainX)
	trainY = np.array(trainY)
	testX = np.array(testX)
	testY = np.array(testY)
    
def initPop():
   
    
    for i in range(POP_SIZE):
        universe = [random() for i in range(MAX_FEATURE)]
        Population.append(universe)
        
    
def calc_Fitness(x):
    
    
    global trainX , trainY, testX, testY
    trTrainX = trainX
    trTestX = testX
	
    trTrainX = np.array(trTrainX)
    trTestX = np.array(trTestX)
	
    ct = 0
    lx = len(x)
    for i in range(lx):
        
        
        if x[i] < 0.5:
            
            trTrainX = np.delete(trTrainX, i-ct,1)
            trTestX = np.delete(trTestX, i-ct, 1)
            ct += 1
            
        
                
        
        
                                
        
    if trTrainX.shape[1] == 0:   # if number of columns == 0
        return 0.0
    clf = KNeighborsClassifier()
	
    clf.fit(trTrainX,trainY)
	
    predicted = clf.predict(trTestX)
		
    ans =  f1_score(testY, predicted, average = 'binary') 
	#print 'ans = %f' % ans*100
	#p.fitness = ans*100
    return ans*100

def allFeatures():
    global trainX , trainY, testX, testY
    trTrainX = trainX
    trTestX = testX
	
    trTrainX = np.array(trTrainX)
    trTestX = np.array(trTestX)

    clf = KNeighborsClassifier()
    print(trTrainX.shape)
	
    clf.fit(trTrainX,trainY)
	
    predicted = clf.predict(trTestX)
		
    ans =  f1_score(testY, predicted, average = 'binary') 
	#print 'ans = %f' % ans*100
	#p.fitness = ans*100
    print("Value with all features{}".format(ans*100))
	
        

def best_cost():
    seq = [x['cost'] for x in Universes]
    return(max(seq))
    
def roulette_wheel_selection(rates):
    fitness_sum = sum(rates)
    probability_offset = 0
    probabilities = []

    for i in range(len(rates)):
        probabilities.append(probability_offset + (rates[i] / fitness_sum))
        probability_offset += probabilities[i]

    r = random()

    selected_ind = 0 # initialize
    for i in range(len(rates)):
        if probabilities[i] > r:
            break; 
        selected_ind = i
    return (selected_ind)


def mvoParkinson():
    readData()
    allFeatures()
    initPop()
    for i in range(len(Population)):
        universeObj = {'universe':Population[i],'cost':calc_Fitness(Population[i])}
        Universes.append(universeObj)
        
    Time = 1
    while Time<MAX_ITER+1:
        
        #WEP Update
        WEP=WEP_Min+Time*((WEP_Max-WEP_Min)/MAX_ITER)
    
        #TDR Update
        TDR=1-((Time)**(1/6)/(MAX_ITER)**(1/6))
        
        BestCost = best_cost()
        BestUniverse = [x['universe'] for x in Universes if x['cost']==BestCost]
        
        
        SortedUniverses = Universes[:]
        SortedUniverses.sort(key = operator.itemgetter('cost'),reverse = True)
        NormalizedRates  = [x['cost'] for x in SortedUniverses]
        NormalizedRates = np.array(NormalizedRates)
        NormalizedRates = (NormalizedRates-np.min(NormalizedRates))/(np.max(NormalizedRates)-np.min(NormalizedRates))
        
        
        for i in range(1,len(Population)):
            black_hole_index = i
            for j in range(MAX_FEATURE):
                #Exploration
                r1 = random()
                if r1<NormalizedRates[i]:
                    white_hole_index = roulette_wheel_selection(NormalizedRates)
                    
                    if white_hole_index ==-1:
                        white_hole_index=0
                    Universes[black_hole_index]['universe'][j]= SortedUniverses[white_hole_index]['universe'][j]
                 
                #Exploitation
                r2 = random()
                if r2<WEP:
                    r3 = random()
                    if r3<0.5:
                        Universes[i]['universe'][j] = BestUniverse[0][j] + TDR*(random())
                    else:
                        Universes[i]['universe'][j] = BestUniverse[0][j] - TDR*(random())
                        
                    if Universes[i]['universe'][j]>1:
                        Universes[i]['universe'][j]=1
                    if Universes[i]['universe'][j]<0:
                        Universes[i]['universe'][j]=0
                    
                    
                
            
        
        
        for i in range(len(Population)):
            Universes[i] = {'universe':Universes[i]['universe'],'cost':calc_Fitness(Universes[i]['universe'])}
            
        print(BestCost)
        
    
        
        Time = Time + 1
        
    
def main():
        
        start_time = time.time()
        mvoParkinson()
        print("****** %s second ******" % (time.time() - start_time))
	
        return 0

if __name__ == '__main__':
	main()
