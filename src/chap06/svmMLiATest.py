# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:33:15 2018

@author: Administrator
"""

import svmMLiA
import numpy as np
# test loadDataSet and relative functions
'''
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
print(dataArr)
print('########################################')
print(labelArr)
'''

# test the basic complete SMO
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
b, alphas = svmMLiA.smop(dataArr, labelArr, 0.6, 0.001, 40)
ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
print("ws is: ", ws)
dataMat = np.mat(dataArr) # classify
numOfData = dataMat.shape[0]
for i in range(numOfData):
    calcLabel = dataMat[i] * np.mat(ws) + b
    print("calculated label:", calcLabel, "| real label: ", labelArr[i])


