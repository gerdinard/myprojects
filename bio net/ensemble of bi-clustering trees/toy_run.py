# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 00:15:32 2021

@author: u0106589
"""

import numpy as np
from Extrabictree import build_bd as ebuild_bd
from Extrabictree import bdtrain as ebdtrain
from Extrabictree import bdtest as ebdtest

############################################################################### 

msleaf = 2
cs = 2
nof_trees = 100

# Dummy random matrices
X1train = np.random.rand(30,10)
X1test = np.random.rand(10,10)
X2 = np.random.rand(20,12)
Ytrain = np.random.randint(2,size=(30,20))
Ytest = np.random.randint(2,size=(10,20))

predtest2 = 0
for i in range(100):
    print(i)
    chleft,chright,features2,thlist,featurelist,impuritylist,nodelist1 = ebuild_bd(X1train,X2,Ytrain,msleaf,cs) # building a randomized bi-clustering tree
    leafnode,leafnode2,prednode,pred,indprednode,pred_rows,pred_cols = ebdtrain(X1train,X2,chleft,chright,features2,thlist,Ytrain) 
    leafnodetest,predtest,Ytest_vector = ebdtest(X1test,X2,chleft,chright,features2,thlist,indprednode,prednode,Ytest,pred_rows,pred_cols) # Ytest is NOT used in the predictions! 
    predtest2 = predtest2 + predtest

predtest = predtest2 / 100.0     

