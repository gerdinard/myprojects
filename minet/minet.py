# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:32:58 2019

@author: Konstantinos Pliakos


License: BSD 3 clause

Implementation of the MINET algorithm [1].


The code below receives as input a dataset that represents a heterogeneous (bi-partite) network and generates a 
global network feature representation.

More specifically, it receives two feature matrices that correspond to the two node-sets of the network. 
In case of supervised learning an interaction matrix is also used as unput. 

Next, two local models are trained, generating two local feature representations. 
The final global network representation is generated as the Cartesian product of the local ones. 
 

Parameters
----------
X1 : matrix of shape = [n_samples, n_features]
(i.e., first feature matrix)

X2 : matrix of shape = [n_samples, n_features]
(i.e., second feature matrix)
        
Y : matrix of shape = [n_samples, n_outputs]
(i.e., the label/output matrix) In case of unsupervised learning, nothing is given as input.


n_est: the number of trees (default=100)

stop_crit: the minimum_samples_per_leaf stopping criterion used (default=5)

p: A float number that controls the node filtering rocess. The nodes containing more than p*n_samples are discarded. 



[1] Pliakos, K., & Vens, C. (2018). Network representation with clustering tree features. 
Journal of Intelligent Information Systems, 51, 2, 341-365.

"""

from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
import numpy as np
from sklearn.decomposition import PCA


class MINET():
    def __init__(self,n_est=200,stop_crit=5,dw=0.9,dim=2,method= 'extra'):
        self.n_est = n_est # number of trees
        self.stop_crit = stop_crit # tree stop.criterion
        self.dw = dw # factor used in node filtering
        self.dim = dim # number of components to be kept in dimensionality reduction
#        self.learning = 'unsupervised'
        self.method = method


    def global_repr(self,X1,X2):
        """Constructing the Cartesian product
    
        Parameters
        ----------
        X1 : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)

        X2 : matrix of shape = [n_samples, n_features]
        (i.e., the second feature matrix)
        
        Returns
        -------
        Xs : The generated Cartesian product
        """
        
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        count = 0
        Xs = np.zeros([N1*N2, X1.shape[1] + X2.shape[1]])
        for i in range(N1):
            for j in range(N2):
                Xs[count] = np.concatenate((X1[i],X2[j]))
                count += 1
        return Xs
    

    def fit_local(self, X, Y=None):
        """Fitting and generating the local space.
    
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)

        Y : matrix of shape = [n_samples, n_outputs]
        (i.e., the label/output matrix)
        """
        
        if self.method == 'rf':
            local = RandomForestRegressor(n_estimators=self.n_est,max_features='sqrt',max_depth=None, min_samples_leaf=self.stop_crit,random_state=0)
            print("Basic model: Random Forest \n")
        else:
            local = ExtraTreesRegressor(n_estimators=self.n_est,max_features='sqrt',max_depth=None, min_samples_leaf=self.stop_crit,random_state=0)
            print("Basic model: Extremely Randomized Trees \n")

        if Y is None:
            local.fit(X,X)
            print("Unsupervised learning \n")
        else:
            local.fit(X,Y)
            print("Supervised learning \n")
            
        treepath = local.decision_path(X)[0]
        w = treepath.sum(0)
        wlog = np.log(w.astype(float))+0.00001            
        local.cw =  np.power(wlog,-1)            
        treepath = treepath.multiply(local.cw).toarray().astype(float)
#        treepath = treepath.toarray().astype(float)
        local.ind = np.where(w<(X.shape[0]*self.dw))[1]
#        treepath = np.delete(treepath,local.ind,axis=1)
        treepath = treepath[:,local.ind]
        
        local.pca = PCA(self.dim)
        local.treepath = local.pca.fit_transform(treepath)
        
        return local          
    
            
    def local_transform(self, local, Xtest):
        """Fitting and generating the local space for the new data.
    
        Parameters
        ----------
        Xtest : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)

        Y : matrix of shape = [n_samples, n_outputs]
        (i.e., the label/output matrix)
        """

            
        treepathtest = local.decision_path(Xtest)[0]
        treepathtest = treepathtest.multiply(local.cw).toarray().astype(float)
#        treepathtest = np.delete(treepathtest,local.ind,axis=1)
        treepathtest = treepathtest[:,local.ind]
        
        treepathtest = local.pca.transform(treepathtest)  

        return treepathtest          


    def fit_transform(self, X1, X2, Y=None):
        """Fitting and generating the MINET space.  
        
        Parameters
        ----------
        X1 : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)

        X2 : matrix of shape = [n_samples, n_features]
        (i.e., the second feature matrix)
                
        Y : matrix of shape = [n_samples, n_outputs]
        (i.e., the label/output matrix)
        
        Returns
        -------
        self.Xg : The generated global network representation
        """
        
        if Y is None:
            self.local1 = self.fit_local(X1)
            self.local2 = self.fit_local(X2)
        else:
            self.local1 = self.fit_local(X1, Y)
            self.local2 = self.fit_local(X2, Y.T)
        
        self.Xg = self.global_repr(self.local1.treepath,self.local2.treepath)           
        
        return self.Xg


    
    def transform(self, Xtest1, Xtest2):
        """Using the fitted model to generating features for new data.  
        
        Parameters
        ----------
        Xtest1 : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)

        Xtest2 : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)
        
        Returns
        -------
        self.Xgtest : The generated global network representation
        """

        treepathtest1 = self.local_transform(self.local1,Xtest1)
        treepathtest2 = self.local_transform(self.local2,Xtest2)
        Xgtest = self.global_repr(treepathtest1,treepathtest2)           
        
        return Xgtest
        
        