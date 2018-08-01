# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:43:33 2018

@author: Administrator
"""


import trees

myDat, labels = trees.createDataSet()
print(myDat)
print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))
