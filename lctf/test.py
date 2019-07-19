# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:57:03 2019

@author: Konstantinos Pliakos

Toy problem for testing LCTF

"""


#from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
#from scipy.sparse import hstack
from mpl_toolkits import mplot3d

#X1 = np.loadtxt('data/Xmedical.txt',delimiter=',').astype('float32')
#Y = np.loadtxt('data/Ymedical.txt',delimiter=',').astype('float32')
#X1 = np.loadtxt('data/Xyeast.txt',delimiter=',').astype('float32')
#Y = np.loadtxt('data/Yyeast.txt',delimiter=',').astype('float32')
#X1 = np.loadtxt('data/mn/X.txt',delimiter=',').astype('float32')
#Y = np.loadtxt('data/mn/Y.txt',delimiter=',').astype('float32')
#X1 = np.loadtxt('data/ppi/X.txt',delimiter=',').astype('float32')
#Y = np.loadtxt('data/ppi/Y.txt',delimiter=',').astype('float32')
X1 = np.loadtxt('data/ern/X1.txt',delimiter=',').astype('float32')
Y = np.loadtxt('data/ern/Y.txt',delimiter=',').astype('float32')


import random
random.seed(0)
train = random.sample(range(X1.shape[0]), int(0.7*X1.shape[0]))
test = list(set(range(X1.shape[0])) - set(train))

Xtrain = X1[train]
Xtest = X1[test]
Ytrain = Y[train]
Ytest = Y[test]


from lctf import LCTF

l = LCTF(n_est=200,stop_crit=5,dw=0.9,dim=3)
treetrain = l.fit_transform(Xtrain,Ytrain)
treetest = l.transform(Xtest)


fig = plt.figure(figsize=(16,40))
ax = fig.add_subplot(8,2,1)
ax.scatter(treetrain[:, 0], treetrain[:, 1],s=8)
ax.set_title("LCTF")

ax = fig.add_subplot(8,2,2, projection ='3d')
ax.scatter(treetrain[:, 0], treetrain[:, 1], treetrain[:, 2],s=8)
ax.set_title("LCTF (3D)")

