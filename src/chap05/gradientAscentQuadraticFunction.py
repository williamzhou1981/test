# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:28:19 2018
以二次函数为例，初步的了解梯度上升算法的优化过程
理论上讲，二次函数可以直接求最值
实际上很多函数无法简单的求最值，只能梯度逼近
这里采用二次函数作为示例，演示逐步逼近的过程
f(x) = -x^2 + 4x
这个函数最大值的时候x为2
@author: Administrator
"""

def gradient_ascend_test():
    def f_prime(x_old):
        return -2 * x_old + 4                    #函数的导数，要是不知道导数咋办
    x_old = -1                                  #初始值，给出一个小于x_new的值
    x_new = 0                                   #梯度上升算法的初始值，即从(0,0)开始
    alpha = 0.01                                #步长，即学习速率，控制更新的频率
    pression = 0.00000001                       #精度，即更新阈值
    while abs(x_new - x_old) > pression:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)
