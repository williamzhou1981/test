# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:53:08 2018

@author: Administrator
"""

import kNN

datingDataMat, datingLabels  = kNN.file2matrix('datingTestSet2.txt') 
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(normMat)

