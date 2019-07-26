# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:23:10 2019

@author: Konstantinos Pliakos

License: BSD 3 clause

Implementation of the LCTF algorithm [1].


The code below receives as input a dataset and generates a 
Low dimensional Clustering Tree Feature (LCTF) representation.


Parameters
----------
X : matrix of shape = [n_samples, n_features]
(i.e., the feature matrix)
        
Y : matrix of shape = [n_samples, n_outputs]
(i.e., the label/output matrix)

n_est: the number of trees (default=100)

stop_crit: the minimum_samples_per_leaf stopping criterion used (default=5)

p: A float number that controls the node filtering process. The nodes containing more than p*n_samples are discarder. 



[1] Pliakos, K., & Vens, C. (2018). Mining features for biomedical data using 
clustering tree ensembles. Journal of biomedical informatics, 85, 40-48.

"""

from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.decomposition import PCA


class LCTF():
    def __init__(self,n_est=200,stop_crit=5,dw=0.9,dim=2):
        self.n_est = n_est # number of trees
        self.stop_crit = stop_crit # tree stop.criterion
        self.dw = dw # factor used in node filtering
        self.dim = dim # number of components to be kept in dimensionality reduction



    def fit_transform(self, X, Y):
        """Fitting and generating the LCTF space.  
        
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)
                
        Y : matrix of shape = [n_samples, n_outputs]
        (i.e., the label/output matrix)
        
        Returns
        -------
        self.treepath : The generated feature representation
        """
        
        self.clf = ExtraTreesRegressor(n_estimators=self.n_est,max_features='sqrt',max_depth=None, min_samples_leaf=self.stop_crit,random_state=0)
        self.clf.fit(X,Y)
        self.treepath = self.clf.decision_path(X)[0]
        
        w = self.treepath.sum(0)
        wlog = np.log(w.astype(float))+0.00001
        self.cw =  np.power(wlog,-1)
        self.treepath = self.treepath.multiply(self.cw).toarray().astype(float)
        self.ind = np.where(w>(X.shape[0]*self.dw))[1]
        self.treepath = np.delete(self.treepath,self.ind,axis=1)
        
        self.pca = PCA(self.dim)
        self.treepath = self.pca.fit_transform(self.treepath)
        
        return self.treepath


    
    def transform(self, Xtest):
        """Using the fitted model to generate features for new data.  
        
        Parameters
        ----------
        Xtest : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)

        Returns
        -------
        self.treepathtest : The generated feature representation
        """
        
        self.treepathtest = self.clf.decision_path(Xtest)[0]
        self.treepathtest = self.treepathtest.multiply(self.cw).toarray().astype(float)
        self.treepathtest = np.delete(self.treepathtest,self.ind,axis=1)
        
        self.treepathtest = self.pca.transform(self.treepathtest)     
        
        return self.treepathtest
        
        
