# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:35:07 2018

@author: Administrator
"""

import kNN

group, labels = kNN.createDataSet()
print(group)
print(labels)

print(kNN.classify0([1,0.5],group,labels,3))