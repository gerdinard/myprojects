# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:54:46 2019

@author: Konstantinos Pliakos

Toy problem for testing MINET

"""

import numpy as np


X1 = np.loadtxt('ern/X1.txt',delimiter=',').astype('float32')
X2 = np.loadtxt('ern/X2.txt',delimiter=',').astype('float32')
Y = np.loadtxt('ern/Y.txt',delimiter=',').astype('float32')


import random
random.seed(0)
train = random.sample(range(X1.shape[0]), int(0.7*X1.shape[0]))
test = list(set(range(X1.shape[0])) - set(train))

import random
random.seed(0)
train2 = random.sample(range(X2.shape[0]), int(0.7*X2.shape[0]))
test2 = list(set(range(X2.shape[0])) - set(train))



Xtrain = X1[train]
Xtest = X1[test]
Xtrain2 = X2[train2]
Xtest2 = X1[test2]
Ytrain = Y[train][:,train2]
Ytest = Y[test][:,test2]


from minet import MINET

m = MINET(n_est=200,stop_crit=5,dw=0.9,dim=2,method='extra')
treetrain = m.fit_transform(Xtrain,Xtrain2,Ytrain)
treetest = m.transform(Xtest,Xtest2)

