# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:17:56 2018

@author: Administrator
"""


import trees
myDat, labels = trees.createDataSet()
print(trees.chooseBestFeatureToSplit(myDat))