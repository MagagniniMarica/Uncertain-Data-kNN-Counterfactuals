# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:05:14 2026

@author: magag
"""
from ucimlrepo import fetch_ucirepo
import copy
import pandas as pd

from sklearn import preprocessing

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
    
    
def data_AD():
     
  
    # fetch dataset 
    adult_ = fetch_ucirepo(id=2) 
  
    # data (as pandas dataframes) 
    X = adult_.data.features 
    y = adult_.data.targets 
  
    # metadata 
    print(adult_.metadata) 
  
    # variable information 
    print(adult_.variables) 
    
    
    
    # Drop columns and missing values
    X = X.drop('education-num', axis=1)
    X = X.drop('relationship', axis=1)
    
    mask0 = X.notna().all(axis=1)
    X,y = X[mask0],y[mask0]
    
    mask1 = X['workclass'] != '?'
    X,y = X[mask1],y[mask1]
    
    mask2 = X['occupation'] != '?'
    X,y = X[mask2],y[mask2]
    
    mask2 = X['native-country'] != '?'
    X,y = X[mask2],y[mask2]
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Target 
    target  = [1 if yn == ">50K" else 0 for yn in y['income'] ] 
    
    
    # Categorical variables
    for j in X.columns:
        if X[j].dtype == 'O':
            X[j] = X[j].astype('category')
            
    # Encode categorical features (ONE HOT ENCODING or ordinal for binary)
    # Initialize the OneHotEncoder
    adult = copy.deepcopy(X) 
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
            adult = adult.drop(j, axis=1)
            adult = pd.concat([adult, encoded_tranformed], axis=1)
            
    #Normalization the non categorical feature
    min_max_scaler = preprocessing.MinMaxScaler()
    for j in adult.columns:
        if adult[j].dtype != 'category':
            print(adult[j].dtype)
            adult.loc[:, j]= min_max_scaler.fit_transform(adult[[j]])
    
    
    
    
    
    # Display info
    adult.keys()
    adult.info()
    adult.describe()
    features=adult.columns              # names of columns vector (Index)
    features_type=feature_type(adult)   # type of each feature vector (Dict)
    
    return adult, target, features, features_type, features_multicat

