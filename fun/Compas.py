# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
"""

#import numpy as np
import pandas as pd
import copy
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


# This function returs:
    # compas: the features DataDrame
    # target: the target Series
    # fetaures: the name of features Index
    # fetaures_type: the type of features Dict
def data_CP(path):
    
    
    full_path =path+'compas-scores-two-years.csv'
    raw_data = pd.read_csv(full_path)
    #print('Num rows: %d' %len(raw_data))
    
    df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & 
          (raw_data['days_b_screening_arrest'] >= -30) &
          (raw_data['is_recid'] != -1) &
          (raw_data['c_charge_degree'] != 'O') & 
          (raw_data['score_text'] != 'N/A')
         )]
    
    df = df.reset_index(drop=True)
    #print('Num rows filtered: %d' % len(df))
    
    X = pd.concat([df['c_charge_degree'], df['age_cat'],df['race'],df['sex'],
                       df['priors_count'], df['decile_score']],axis=1)
    
    #Categorical variables
    for f in ['c_charge_degree', 'age_cat', 'race', 'sex']:
        X[f]=X[f].astype('category') 
    
    #Integer variables
    X['priors_count'] = X['priors_count'].astype('int64') 
    X['decile_score'] = X['decile_score'].astype('int64') 
    
    #Target
    target = df['two_year_recid'].apply(lambda x: 1 if x==0 else 0) #positive class 1 is no recid

    
    # Encode categorical features (ONE HOT ENCODING or ordinal for binary)
    # Initialize the OneHotEncoder
    compas = copy.deepcopy(X) 
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
                encoded_tranformed = pd.DataFrame(encoded, columns=encoder_OH.get_feature_names_out([j]), dtype='category')
                features_multicat[j] = encoded_tranformed.columns
            compas = compas.drop(j, axis=1)
            compas = pd.concat([compas, encoded_tranformed], axis=1)
             
           
    #Normalization the non categorical feature
    min_max_scaler = preprocessing.MinMaxScaler()
    for j in compas.columns:
        if compas[j].dtype != 'category':
            print(compas[j].dtype)
            compas.loc[:, j]= min_max_scaler.fit_transform(compas[[j]])
     
    




    # Display info
    compas.keys()
    compas.info()
    compas.describe()
    features=compas.columns              # names of columns vector (Index)
    features_type=feature_type(compas)   # type of each feature vector (Dict)

    
    return compas, target, features, features_type, features_multicat
    