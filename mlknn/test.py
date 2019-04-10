# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:17:30 2019

@author: Konstantinos Pliakos

Toy problem for testing
"""

#from sklearn.neighbors import NearestNeighbors
from mlknn import MLKNN
import numpy as np

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




# Call class mlknn
ml = MLKNN()
ml.fit(Xtrain, Ytrain, 5)
Pre_Labels,Outputs = ml.predict(Xtest,Ytrain,5)



# Test external kNN
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(5,n_jobs=1)
neigh.fit(Xtrain)
Neighbors = neigh.kneighbors(Xtrain, 5, return_distance=False)
Neighbors_test = neigh.kneighbors(Xtest, 5, return_distance=False)


ml2 = MLKNN()
ml2.fit(Xtrain, Ytrain, 5, neighbors = Neighbors)
Pre_Labels2,Outputs2 = ml2.predict(Xtest,Ytrain,5,neighbors = Neighbors_test)



# Test using own distance matrix
from sklearn.metrics.pairwise import euclidean_distances
Xtraind = euclidean_distances(Xtrain,Xtrain)
Xtestd = euclidean_distances(Xtest,Xtrain)


ml3 = MLKNN()
ml3.fit(Xtraind, Ytrain, 5, distance_matrix = True)
Pre_Labels3,Outputs3 = ml3.predict(Xtestd,Ytrain,5,distance_matrix_t = True)















