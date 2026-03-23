# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:56:13 2024

@author: Maric
"""


import pandas as pd
from sklearn import preprocessing
from ucimlrepo import fetch_ucirepo 


# This function returns a dict, for each feature is associated 
# the type (categorical (also binary), integer or numerical (continuaos))
def feature_type(dataset):
        features=list(dataset.columns)
        feat_type =[]
        for x in dataset.dtypes:
            if x.name == 'category':
                feat_type.append('Categorical')
            else:
                feat_type.append('Numerical')                
        features_type = dict(zip(features, feat_type))
        return features_type
    
# This function returs:
    # bcwd: the features DataDrame
    # target: the target Series
    # fetaures: the name of features Index
    # fetaures_type: the type of features Dict
def data():
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=15) 
    bcwd = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # metadata 
    print(breast_cancer_wisconsin_diagnostic.metadata) 
  
    # variable information 
    print(breast_cancer_wisconsin_diagnostic.variables) 
     
  
     
    mask0 = bcwd.notna().all(axis=1)
    bcwd,y = bcwd[mask0],y[mask0]
    
    
    # Target 
    target  = [1 if yn == "yes" else 0 for yn in y['y'] ] 
    
   
    #Normalization the non categorical feature
    min_max_scaler = preprocessing.MinMaxScaler()
    for j in bcwd.columns:
        if bcwd[j].dtype != 'category':
            print(bcwd[j].dtype)
            bcwd.loc[:, j]= min_max_scaler.fit_transform(bcwd[[j]])
    
    features_multicat = {}
   
    # Display info
    bcwd.keys()
    bcwd.info()
    bcwd.describe()
    features=bcwd.columns              # names of columns vector (Index)
    features_type=feature_type(bcwd)   # type of each feature vector (Dict)
   
    
    return bcwd, target, features, features_type, features_multicat
        