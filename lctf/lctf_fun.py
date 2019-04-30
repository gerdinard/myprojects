# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:00:18 2019

@author: Konstantinos Pliakos

License: BSD 3 clause

Implementation of the LCTF algorithm [1].


The function below receives as input a dataset and its train/test indices. 
It ouputs a Low dimensional Clustering Tree Feature (LCTF) representation.


Parameters
----------
X : matrix of shape = [n_samples, n_features]
(i.e., the feature matrix)
        
Y : matrix of shape = [n_samples, n_outputs]
(i.e., the label/output matrix)

train: the training instances
test: the test instances

n_est: the number of trees (default=100)

stop_crit: the minimum_samples_per_leaf stopping criterion used (default=5)

p: A float number that controls the node filtering rocess. The nodes containing more than p*n_samples are discarder. 



[1] Pliakos, K., & Vens, C. (2018). Mining features for biomedical data using 
clustering tree ensembles. Journal of biomedical informatics, 85, 40-48.

"""

from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.decomposition import PCA


#X1 = np.loadtxt('Xmedical.txt',delimiter=',').astype('float32')
#Y = np.loadtxt('Ymedical.txt',delimiter=',').astype('float32')
X1 = np.loadtxt('Xyeast.txt',delimiter=',').astype('float32')
Y = np.loadtxt('Yyeast.txt',delimiter=',').astype('float32')


import random
random.seed(0)
train = random.sample(range(X1.shape[0]), int(0.7*X1.shape[0]))
test = list(set(range(X1.shape[0])) - set(train))

Xtrain = X1[train]
Xtest = X1[test]
Ytrain = Y[train]
Ytest = Y[test]



#LCTF
def lctf_fun(X,Y,train,test,n_est=200,stop_crit=5,dw=0.9,dim=2):  
   
    
    rfobj = ExtraTreesRegressor(n_estimators=n_est,max_features='sqrt',max_depth=None, min_samples_leaf=stop_crit,random_state=0)
    rfobj.fit(X[train],Y[train])
    treepath = rfobj.decision_path(X)[0]

    w = treepath[train].sum(0)
    wlog = np.log(w.astype(float))+0.00001
    cw =  np.power(wlog,-1)
    treepath = treepath.multiply(cw)
    treepath = treepath.toarray().astype(float)
    treepath = np.delete(treepath,np.where(w>(len(train)*dw))[1],axis=1)

    pca = PCA(dim)
    treepathtr = pca.fit_transform(treepath[train])
    treepathtest = pca.transform(treepath[test])

    return treepathtr,treepathtest,rfobj


treepathtr,treepathtest,rfobj = lctf_fun(X1,Y,train,test,n_est=200,stop_crit=5,dw=0.9,dim=2)


#################################################











