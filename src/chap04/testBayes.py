# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:19:20 2018

@author: Administrator
"""


import bayes
#bayes.testingNB() # 测试spamTest的时候注释掉而已

'''
    最好进行多次测试
    少量测试会得出errorCount为零，整个值不真实
    多次测试后取个算术平均值就可以了
'''
for i in range(10):
    bayes.spamTest()