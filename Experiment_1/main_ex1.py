# -*- coding: utf-8 -*-
"""
@author: Marica 

Experiment 1

"""

#
# Labraries, packages, classes
#
from fun import GroundSyntheticDataset_ , SymbolicDataset_,Symbolic_US
from fun import d_label_sensitive,  kNN_rule, exact_COP_
from fun import save_txt_, save_exel_
from Experiment_1.plots import plot_data, plot_count_,plot_compare_count_
import numpy as np
import pandas as pd
from pathlib import Path
import os
import json

User = 'Maric' #'Maric' 'magag'


#%% Problem and dataset Parameters

# Folder 
save = True
input_ = 'Configuration 1'
# path  Experiment_1
if '__file__' in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path(os.getcwd())

seed = 10                               # Random generation seed

DataType = "B"                          #"R" rectangles or "B" balls
DataTypes =["B"]#, "R"]

#
# Distance Type
#
distances = ['d0','d1','dint2']         # 'd0' : all min dist, 'd1' or 'dint2': max  or int2 dist for label 1
d_flag = distances[0]

#
# Counterfactual problem
#
K = [3,5,7]                             # Elements in the neigborhood
M =  1e3                                # Big-M 
alpha = 1e-1*5                          # Probability of the classifier
eps = 1e-5                              # Nk construction

k=3

#
# Sythetic dataset
#

name_dataset = 'Synthetic'
#Synthetic Dataset of 
N = 30                                  # Number of elements in the dataset
J = 2                                   # Number of features
mean_std_0 = {'mean' : 2, 'std' : np.sqrt(64)}
mean_std_1 = {'mean' : 10, 'std' : np.sqrt(64)}
ground_dataset, target, features, features_type = GroundSyntheticDataset_(seed, J, N, mean_std_0,mean_std_1)
features_multicat = {}

#
# Symbolic Dataset
#

# percentuale incertezza dataset. Esempio: fra il 5% e il 10% ---> [0.05,0.1]
pc_Un_range = (0.04,0.09)  
pc_U,pc_S = (0.04, 0.04)         # percentuale incertezza input e controfattuale
Uc = {'x1': -8.566964377192644, 'x2': -1.0290210615433626}           # Input center
Sc = {f:0 for f in features}    # Counterafactual shape center  


#
# Solver
#
timelim = 100
###############################################################################
# Save all experiment details in .txt
###############################################################################
params = {
# General
"seed (random)": seed,
"DataType": DataType,
# Distances
"Notion of distance": d_flag,

# Counterfactual problem
"k": k,
"M": M,
"alpha": alpha,
"eps": eps,

# Input U, 
"U center": Uc,
# Counterfactual S+x, S shape x Unknown
"S center": Sc,

# Dataset
'Dataset name': 'Synthetic',
"Features": features,
"Number of elements (N)": N,
"Number of features (J)": J,
"Uncertainty percentage range data": pc_Un_range,
"Uncertainty percentage Input": pc_U,
"Uncertainty percentage counterfactual": pc_S,
"0-label distribution": mean_std_0,
"1-label distribution": mean_std_1,

# Solver
"Exact Solver Time Limit (s)": timelim    
}
foldername = f'{d_flag}_k{k}'
    

    
#%%  Series of computations 

Input_DIR = BASE_DIR / input_
Input_DIR.mkdir(exist_ok=True)# crea le cartelle se non esistono

for DataType in DataTypes:
    path = Input_DIR / f"{DataType}"
    path.mkdir(exist_ok=True)        
    params['DataType'] = DataType
    
    #
    # Symblic data definition
    #
    data = SymbolicDataset_(DataType, ground_dataset, target, features,features_type, seed, N,pc_Un_range)
    U,S = Symbolic_US(DataType,ground_dataset, features, features_type,N,
                    Uc,Sc, pc_U,pc_S, )
    # plot dataset
    plot_data(data,U, margin = 0.1 , figsize_cm=15, save = save, path= path, filename = 'dataset')
   
    for k in K:
        params['k'] = k
        # Folder
        foldername = f'k={k}'
               
        for d_flag in distances:
            params['Notion of distance'] = d_flag
            
            #
            # Save options
            #
            if save:
                folderpath = save_txt_(params, path, foldername=foldername)
            else:
                folderpath = path
            
                
            # Check input instance if 0-label
            Ulabel, U_Nk = kNN_rule(k,d_flag, data, U)
            print(f'U label for sensitive distance kNN rule: {Ulabel}, \n Neighbors: {U_Nk}')
            
            #
            # Exact Counterfactual computation
            #
            d_obj = d_flag
            Sx, Nk_index, d_UxS,time_exact = exact_COP_(DataType,d_flag, d_obj,
                                                        data, features, features_type, U, S,N, J,
                                                        k, M, eps, timelim)
            #
            # Plot
            #
            Nk_Sx = {}
            for n in range(N):
                if n in Nk_index:
                    Nk_Sx[data[n].name] = [d_label_sensitive(Sx,data[n], d_flag), data[n].label]

            plot_count_(d_flag,U, Sx,data,Nk_Sx,
                          margin = 0.1 , figsize_cm=15, save = True,
                          path= folderpath, filename = f'Exact_{d_flag}')
           
            #
            # Save
            #

            results= {'xS': Sx.x, f'N_{k}' : list(Nk_Sx.keys()),  "distance type" : d_flag, "distance U-x+S": d_UxS, "Time Execution" : time_exact}
            if save:
                save_exel_('Exact', results, path, foldername = folderpath)
                
                
#%% Plot Comparisons



Input_DIR = BASE_DIR / input_

for DataType in DataTypes:
    path = Input_DIR / f"{DataType}"
    
    #
    # Symblic data definition
    #
    data = SymbolicDataset_(DataType, ground_dataset, target, features,features_type, seed, N,pc_Un_range)
    U,S = Symbolic_US(DataType,ground_dataset, features, features_type,N,
                    Uc,Sc, pc_U,pc_S, )
    
    for k in K:
        folderpath = path / f'k={k}'

        columns = ['xS','distance type','distance U-x+S']
        df = pd.read_excel(Path(folderpath) / "results_Exact.xlsx")
        counterfactuals = df[columns]
        counterfactuals.loc[:, "xS"] = counterfactuals["xS"].apply(json.loads)
        
        plot_compare_count_(U, counterfactuals ,data,
                      margin = 0.1 , figsize_cm=15, save = True,
                      path= folderpath, filename = 'Compare conterfactuals')
