# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:33:03 2026

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
    
    
def data_BM():
     
  
    # fetch dataset 
    bank_marketing = fetch_ucirepo(id=222) 
  
    # data (as pandas dataframes) 
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 
  
    # metadata 
    print(bank_marketing.metadata) 
  
    # variable information 
    print(bank_marketing.variables) 
     
    # Drop columns and missing values
    X = X.drop('day_of_week', axis=1)
    X = X.drop('month', axis=1)
    
    mask0 = X.notna().all(axis=1)
    X,y = X[mask0],y[mask0]
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # Target 
    target  = [1 if yn == "yes" else 0 for yn in y['y'] ] 
    
    # Categorical variables
    for j in X.columns:
        if X[j].dtype == 'O':
            X[j] = X[j].astype('category')
            
    # Encode categorical features (ONE HOT ENCODING or ordinal for binary)
    # Initialize the OneHotEncoder
    bank_marketing = copy.deepcopy(X) 
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
            bank_marketing = bank_marketing.drop(j, axis=1)
            bank_marketing = pd.concat([bank_marketing, encoded_tranformed], axis=1)
            
    #Normalization the non categorical feature
    min_max_scaler = preprocessing.MinMaxScaler()
    for j in bank_marketing.columns:
        if bank_marketing[j].dtype != 'category':
            print(bank_marketing[j].dtype)
            bank_marketing.loc[:, j]= min_max_scaler.fit_transform(bank_marketing[[j]])
    
    
    
    
    
    # Display info
    bank_marketing.keys()
    bank_marketing.info()
    bank_marketing.describe()
    features=bank_marketing.columns              # names of columns vector (Index)
    features_type=feature_type(bank_marketing)   # type of each feature vector (Dict)
    
    return bank_marketing, target, features, features_type, features_multicat

