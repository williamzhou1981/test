# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:21:09 2018

@author: Administrator
"""

'''
    熵计算方法
'''

from math import log
def clcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        #featVec是个向量，前面字段是特征，最后一个是类别,这里直接获取最后一个类别信息
        currentLabel = featVec[-1] 
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'],[1, 0, 'no'],
               [0, 1, 'no'],[0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

