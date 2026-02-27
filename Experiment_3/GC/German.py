# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:17:20 2025

@author: Maric
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from os.path import join
from sklearn import preprocessing
import copy

# This function returns a dict, for each feature is associated 
# the type (categorical (also binary), integer or numerical (continuaos))
def feature_type(dataset):
        features=list(dataset.columns)
        feat_type =[]
        for x in dataset.dtypes:
            if x.name == 'category':
                feat_type.append('Categorical')
            elif x.name == 'int64':
                feat_type.append('Integer')
            else:
                feat_type.append('Numerical')                
        features_type = dict(zip(features, feat_type))
        return features_type

def data_GC():
    # fetch dataset 
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    # data (as pandas dataframes) 
    X = statlog_german_credit_data.data.features 
    y = statlog_german_credit_data.data.targets 
    
    # Target 
    target  = [yn if yn == 1 else 0 for yn in y['class'] ] # 1: good creditor, 0: bad creditor
    
    # Set categorical features
    for j in X.columns:
        if X[j].dtype == 'O':
            X[j] = X[j].astype('category')
            
     
    # Encode categorical features (ONE HOT ENCODING or ordinal for binary)
    # Initialize the OneHotEncoder
    german = copy.deepcopy(X) 
    features_multicat = {}
    encoder_OH = preprocessing.OneHotEncoder(sparse_output=False)  # sparse=False returns a dense array
    encoder_O  = preprocessing.LabelBinarizer() # single column for binary variables
    for j in X.columns:
        if X[j].dtype == 'category':
            if X[j].nunique() == 2: # binary
                encoded = encoder_O.fit_transform(X[[j]])
                encoded_tranformed = pd.DataFrame(encoded, columns = [j], dtype='category')
            else:    # multiple labels
                encoded = encoder_OH.fit_transform(X[[j]])
                num_j =len(encoder_OH.get_feature_names_out([j])) 
                encoded_tranformed = pd.DataFrame(encoded, columns=[j+f'_{i+1}' for i in range(num_j)], dtype='category')
                features_multicat[j] = encoded_tranformed.columns
            german = german.drop(j, axis=1)
            german = pd.concat([german, encoded_tranformed], axis=1)
            
            
            
            
    
    
    #Normalization the non categorical feature
    min_max_scaler = preprocessing.MinMaxScaler()
    for j in german.columns:
        if german[j].dtype != 'category':
            print(german[j].dtype)
            german.loc[:, j]= min_max_scaler.fit_transform(german[[j]])
    
    
    
    
    
    # Display info
    german.keys()
    german.info()
    german.describe()
    features=german.columns              # names of columns vector (Index)
    features_type=feature_type(german)   # type of each feature vector (Dict)
    
    return german, target, features, features_type, features_multicat



  

  

