# -*- coding: utf-8 -*-
"""
@author: Konstantinos Pliakos
"""


from sklearn.tree import DecisionTreeRegressor
#from sklearn.tree import DecisionTreeClassifier

#from sklearn.cross_validation import StratifiedKFold as skfold
#from sklearn.cross_validation import KFold as kfold

#from sklearn.metrics import roc_auc_score,average_precision_score#,accuracy_score

import numpy as np
#from scipy import sparse


# vectorization of a vector  
# you can also do X.reshape(-1)
def mat_to_vector(X):
    count = 0
    N1 = X.shape[0]
    N2 = X.shape[1]
    Xs = np.empty([N1*N2])
    for i in range(N1):
        for j in range(N2):
            Xs[count] = X[i,j]
            count += 1
    return Xs
            

def rowsplit(clf,parentXr,Yr,N):
    clf.fit(parentXr,Yr)
    feature = clf.tree_.feature[0]
    impurity = clf.tree_.impurity[0]
    thres = clf.tree_.threshold[0]
    idr = clf.apply(parentXr)
    if len(clf.tree_.impurity) == 1:
        impurity = clf.tree_.impurity[0]
    else:
        impurity = (clf.tree_.impurity[0] - clf.tree_.impurity[1]*(float(sum(idr==1))/len(idr)) - clf.tree_.impurity[2]*(float(sum(idr==2))/len(idr)))*(float(parentXr.shape[0])/N)
    return feature,impurity,thres,idr#,varRed // /(Yr.shape[1])
    
    

# This function builds the tree
def build_bd(X1,X2,Y,splr,splc):
    
    Nsamples = np.size(X1,0) # Number of samples (row, col)
    Nsamples_c = np.size(X2,0)    
    
    thlist =[] # Initialization of the three lists containing (thresholds, features, impurity)
    featurelist = []
    impuritylist = []
    
    chleft = np.zeros([2*(Nsamples*Nsamples_c),1]) # this has to change
    chright = np.zeros([2*(Nsamples*Nsamples_c),1]) # set a very high value for initialization purposes 
    branch = [] # list representing every branch of the tree, it contains the nodeid and the left-right sign
    nodelist1 = [] # auxiliary list storing the node id and the row or col sign in order to know if the split was made rowwise or columnwise
    nodelist = [] # probably unnecessary
    
    # one list containing the indices of the row-column samples 
    nodestack = [[np.asarray(range(Nsamples)),np.asarray(range(Nsamples_c))]]   
        
    node_id = 0 #initialization of the node counter
        
###    the whole tree-growing procedure
    while len(nodestack)>0: 
#        print node_id
        parentX1, Yparent = X1[nodestack[-1][0]],Y[nodestack[-1][0],:] # the last item of the list is stored in  temp vars
        Yparent = Yparent[:,nodestack[-1][1]] # here the label matrix is transformed to meet the node-subsample
        
        parentX2, Yparent2 = X2[nodestack[-1][1]],Y[:,nodestack[-1][1]]
        Yparent2 = Yparent2[nodestack[-1][0],:]    
        
        clf = DecisionTreeRegressor(max_depth=1, max_features="sqrt",splitter="random", min_samples_leaf=splr)
        testr = rowsplit(clf,parentX1,Yparent,Nsamples) # two (best) splits are performed columnwise and rowise 

        clf = DecisionTreeRegressor(max_depth=1, max_features="sqrt",splitter="random", min_samples_leaf=splr)
        testc = rowsplit(clf,parentX2,Yparent2.T,Nsamples_c)
        
        if (sum(testr[3]) == 0 and sum(testc[3]) == 0): #check if we have a leaf
            pass
        elif sum(testr[3])==0:
            testr = [-1,-1,-1,-1]
        elif sum(testc[3])==0:
            testc = [-1,-1,-1,-1]
        
        if testr[1] > testc[1]: #the splitting has been performed. The highest impurity reduction wins!! 
            featurelist.append(testr[0])# lists are formed containing the split features, impurity, threshold etc.
            impuritylist.append(testr[1])
            thlist.append(testr[2])
            idx = testr[3]   # the ids of the samples (1-> left child, 2 -> right child)
            nodestack.append([np.asarray(nodestack[-1][0])[idx==2],nodestack[-1][1]]) #right
            nodestack.append([np.asarray(nodestack[-2][0])[idx==1],nodestack[-1][1]]) #left
            nodelist1.append(['row',node_id]) # auxiliary list storing the node id and the row or col sign in order to know if the split was made rowwise or columnwise
        else:
            featurelist.append(testc[0])
            impuritylist.append(testc[1])
            thlist.append(testc[2])
            idx = testc[3]   
            nodestack.append([np.asarray(nodestack[-1][0]),nodestack[-1][1][idx==2]]) #right
            nodestack.append([np.asarray(nodestack[-1][0]),nodestack[-2][1][idx==1]]) #left
            nodelist1.append(['col',node_id])    
          
        branch.append(['right',node_id])
        branch.append(['left',node_id])
        if (node_id>0 and branch[-3][0] is 'left'):
            chleft[branch[-3][1]] = node_id
            branch.pop(-3)
        elif (node_id>0 and branch[-3][0] is 'right'):
            chright[branch[-3][1]] = node_id
            branch.pop(-3)       
        
        if sum(idx)==0: # check if there is no split (i.e., leaves)
            nodestack.pop(-1) # the two added items to the stack
            nodestack.pop(-1)
            nodestack.pop(-1) 
            
            nodelist.append(-1)
            nodelist1[-1] = [-1]
            
            branch.pop(-1)
            branch.pop(-1)
        else:
            nodestack.pop(-3)
            
            nodelist.append(node_id)
        node_id = node_id + 1
       
    # after having built the tree, prune the children lists
    l = (np.asarray(featurelist)==-2)
    chleft = np.delete(chleft,range(len(featurelist),2*(Nsamples*Nsamples_c)),axis=0)
    chright = np.delete(chright,range(len(featurelist),2*(Nsamples*Nsamples_c)),axis=0)   
        
    chleft[l] = -1
    chright[l]= -1    
   
   # set the correct indices of the split features 
#   (i.e., if we have the feature 1 for a column, the right index should be 1 + no_row_features)
    index = []
    for i in nodelist1:
        if i[0] is 'row':
            index.append(0)
        else:
            index.append(1)
            
    index = np.asarray(index)
    index[chleft[:,0]==-1] = 0
    features = np.asarray(featurelist)
    features2 = features + (index*X1.shape[1])
    features2 = features2.tolist()
    
    return chleft,chright,features2,thlist,featurelist,impuritylist,nodelist1
#
#def bdtrain(X1,X2,chleft,chright,features2,thlist,Y):
#    count = 0
#    N1 = X1.shape[0]
#    N2 = X2.shape[0]
#    Nsamples = N1*N2 
#    
#    #apply function
#    leafnode = np.zeros(Nsamples)
#    leafnode2 = np.zeros_like(leafnode)
#    
#    for i in range(N1):
#        for j in range(N2):
#    #        print i,j
#            node = 0 #root
#            Xtemp = np.concatenate((X1[i],X2[j]))
#            while chleft[node][0] != -1:
#    #            print node
#                if Xtemp[features2[node]] <= thlist[node]:
#                    node = chleft[node][0].astype(int)
#                else:
#                    node = chright[node][0].astype(int)
#                
#            leafnode[count] = node #store where each sample goes
#            leafnode2[count] = Y[i,j] #store the label of each sample
#            count += 1
#    l = np.unique(leafnode)
##    print len(l)
##    print len(np.where(chleft==-1)[0])
#        
#    pred = np.zeros(Nsamples)   # predict the values for each sample
#    prednode = np.zeros(len(l)) # compute the values of each node
##    indprednode = np.zeros(len(l))  
#    
#    for count,cont in enumerate (l):
#        prednode[count] = np.mean(leafnode2[leafnode==cont])
##        indprednode[count] = cont # (redundant) the same value as l
#        pred[leafnode==cont] = prednode[count]    
#
#    return leafnode,leafnode2,prednode,pred,l
#    
#    
#def bdtest(X1,X2,chleft,chright,features2,thlist,indprednode,prednode,Ytest):
#    # testing phase
#    count = 0
#    Nsamplestest = X1.shape[0]*X2.shape[0] 
#    #apply
#    leafnodetest = np.zeros(Nsamplestest)
#    predtest = np.zeros(Nsamplestest)
#    
#    for i in range(X1.shape[0]):
#        for j in range(X2.shape[0]):
#    #        print i,j
#            node = 0 #root
#            Xtemp = np.concatenate((X1[i],X2[j]))
#            while chleft[node][0] != -1:
#    #            print node
#                if Xtemp[features2[node]] <= thlist[node]:
#                    node = chleft[node][0].astype(int)
#                else:
#                    node = chright[node][0].astype(int)
#                
#            leafnodetest[count] = node #store where each sample goes
#            count += 1
#    
##    predtest = []
#    for count,i in enumerate(leafnodetest):
#        predtest[count] = prednode[indprednode == i]
##        predtest.append(prednode[indprednode == i])
#        
##    predtest = np.asarray(predtest)
#    
#    Ytest = mat_to_vector(Ytest)
#    
#    return leafnodetest,predtest,Ytest   
    





def bdtrain(X1,X2,chleft,chright,features2,thlist,Y):
    count = 0
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    Nsamples = N1*N2
    
    #apply
    leafnode = np.zeros(Nsamples)
    leafnode2 = np.zeros_like(leafnode)
    #indices of rows and columns
    nodeindex_rows = np.zeros_like(leafnode).astype(int)
    nodeindex_cols = np.zeros_like(leafnode).astype(int)
    
    
    
    for i in range(N1):
        for j in range(N2):
    #        print i,j
            node = 0 #root
            Xtemp = np.concatenate((X1[i],X2[j]))
            while chleft[node][0] != -1:
    #            print node
                if Xtemp[features2[node]] <= thlist[node]:
                    node = chleft[node][0].astype(int)
                else:
                    node = chright[node][0].astype(int)
                
            leafnode[count] = node #store where each sample goes
            leafnode2[count] = Y[i,j] #store the label of each sample
            nodeindex_rows[count] = i
            nodeindex_cols[count] = j
            count += 1
    l = np.unique(leafnode)
#    print len(l)
#    print len(np.where(chleft==-1)[0])
        
    pred = np.zeros(Nsamples)   # predict the values for each sample
    prednode = np.zeros(len(l)) # compute the values of each node
#    indprednode = np.zeros([len(l)])  
    prednode_rows = np.zeros([len(l),Y.shape[1]])    
    prednode_cols = np.zeros([len(l),Y.shape[0]]) 
    
    for count,cont in enumerate (l):
        prednode[count] = np.mean(leafnode2[leafnode==cont]) # the total average
        pred[leafnode==cont] = prednode[count] # the prediction using the total average
        prednode_rows[count] = np.mean(Y[nodeindex_rows[leafnode==cont]],0)
        prednode_cols[count] = np.mean(Y[:,nodeindex_cols[leafnode==cont]],1)
#        indprednode[count] = cont # (redundant) the same value as l
           

    return leafnode,leafnode2,prednode,pred,l,prednode_rows,prednode_cols
    
    
def bdtest(X1,X2,chleft,chright,features2,thlist,indprednode,prednode,Ytest,pred_rows,pred_cols):
    # testing phase
    count = 0
    Nsamplestest = X1.shape[0]*X2.shape[0] 
    
    #apply
    leafnodetest = np.zeros(Nsamplestest)
    nodeindex_rows = np.zeros_like(leafnodetest).astype(int)
    nodeindex_cols = np.zeros_like(leafnodetest).astype(int)
    
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
    #        print i,j
            node = 0 #root
            Xtemp = np.concatenate((X1[i],X2[j]))
            while chleft[node][0] != -1:
    #            print node
                if Xtemp[features2[node]] <= thlist[node]:
                    node = chleft[node][0].astype(int)
                else:
                    node = chright[node][0].astype(int)
                
            leafnodetest[count] = node #store where each sample goes
            nodeindex_rows[count] = i
            nodeindex_cols[count] = j            
            count += 1
    
    predtest = []
    for count,i in enumerate(leafnodetest):
#        predtest.append(prednode[indprednode == i])
        predd = pred_rows[indprednode == i]
        predtest.append(predd[:,nodeindex_cols[count]])
#        predd = pred_cols[indprednode == i]
#        predtest.append(predd[:,nodeindex_rows[count]])
        
        
    predtest = np.asarray(predtest)
    
    Ytest = mat_to_vector(Ytest)
    
    return leafnodetest,predtest,Ytest   
    
    
    
def bdtestc(X1,X2,chleft,chright,features2,thlist,indprednode,prednode,Ytest,pred_rows,pred_cols):
    # testing phase
    count = 0
    Nsamplestest = X1.shape[0]*X2.shape[0] 
    
    #apply
    leafnodetest = np.zeros(Nsamplestest)
    nodeindex_rows = np.zeros_like(leafnodetest).astype(int)
    nodeindex_cols = np.zeros_like(leafnodetest).astype(int)
    
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
    #        print i,j
            node = 0 #root
            Xtemp = np.concatenate((X1[i],X2[j]))
            while chleft[node][0] != -1:
    #            print node
                if Xtemp[features2[node]] <= thlist[node]:
                    node = chleft[node][0].astype(int)
                else:
                    node = chright[node][0].astype(int)
                
            leafnodetest[count] = node #store where each sample goes
            nodeindex_rows[count] = i
            nodeindex_cols[count] = j            
            count += 1
    
    predtest = []
    for count,i in enumerate(leafnodetest):
#        predtest.append(prednode[indprednode == i])
#        predd = pred_rows[indprednode == i]
#        predtest.append(predd[:,nodeindex_cols[count]])
        predd = pred_cols[indprednode == i]
        predtest.append(predd[:,nodeindex_rows[count]])
        
        
    predtest = np.asarray(predtest)
    
    Ytest = mat_to_vector(Ytest)
    
    return leafnodetest,predtest,Ytest     

