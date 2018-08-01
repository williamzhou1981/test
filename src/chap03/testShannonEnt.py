# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:56:28 2018

@author: Administrator
"""

import trees

myDat, labels = trees.createDataSet()
print(myDat)
print(trees.clcShannonEnt(myDat))
