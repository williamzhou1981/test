# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 16:56:08 2018

@author: Administrator
"""

import kNN2_2
datingDataMat, datingLabels = kNN2_2.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(len(datingDataMat))
print(datingLabels)
print(len(datingLabels))
