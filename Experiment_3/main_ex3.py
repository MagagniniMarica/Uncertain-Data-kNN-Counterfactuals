# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
"""

#
# Labraries, packages, classes
#

from pathlib import Path
import pandas as pd

from fun import Dataset_selection, SymbolicDataset_, Symbolic_US
from fun import kNN_rule, end_counterf_, Heuristic_counterf_
from fun import save_txt_, save_exel_

from Experiment_3.plots import heatmap_,  heatmap_input_

User = "" #INSERT

#%% Parameters

# Folder 
save = True

# path  Experiment_3
BASE_DIR = Path(__file__).resolve().parent

seed = 10                                                                       # Random generation seed

name_dataset = 'AD'                                                             
# BH : Boston Housing,GC german credit, CP compas, AD abult income, BCW breast cancer winsconsin, BM bank marketing
          
DataType = "R"                                                                  #"R" rectangles or "B" balls
dataset_DIR = BASE_DIR / name_dataset

distances = ['d0','d1','dint2'] if name_dataset == 'BH' else ['d1']              # 'd0' : all min dist, 'd1' or 'dint2': max  or int2 dist for label 1

# Counterfactual problem
K = [5] if name_dataset in ['BH','BCW'] else [15] if name_dataset in ['CP','BM'] else [9] if name_dataset == 'GC' else [105]  # Elements in the neigborhood
M =  100                                                                        # Big-M 
eps_heur = 1e-5                                                                 # Local costraints
I = 10 if name_dataset == 'BH' else 2                                           # Number of inputs, 10 in BH, 2 otherwise

#
# Symbolic Dataset, U and S configuration
#
#
Uncert_degree = [0, 0.02, 0.05] if name_dataset == 'BH' else [0, 0.02]          # 0%, 2%, 5%

###############################################################################
# Solver-Heuristic params
###############################################################################
timelim = 600                                   # Exact and Heuristic time limit

beta = 0.99                                     # Probability for GaussianVNS perturbations
G = 6                                           # number of sigma components, maxGaussVNS index

solver_opts = {'TimeLimit': 1}



#%% Dataset Initialization

#
# Ground Dataset
#
# Benchmarck dataset
if name_dataset == 'BH':
    ground_dataset, target, features, features_type = Dataset_selection(name_dataset)
    features_multicat = {} 
else:
    ground_dataset, target, features, features_type, features_multicat = Dataset_selection(name_dataset)
N,J = ground_dataset.shape


#
# Input U, Counterfactual S+x 
#
Sc = {f:0 for f in features}                                # center


"""
I_= []
Uc_set = []
n = -1
while n < N and len(I_) < I:
    n+=1
    Uc =  dict(ground_dataset.iloc[n])
    bad = False
    print(f"{n}")
    
    for pc_U in Uncert_degree:
        if bad: break
        # percentuale incertezza dataset. Esempio: fra il 5% e il 10% ---> [0.05,0.1]
        pc_Un_range = (pc_U,pc_U)  
        for pc_S in Uncert_degree:
            if bad: break
            #
            # Symblic data definition
            #
            data = SymbolicDataset_(DataType, ground_dataset, target, 
                                    features,features_type, seed, N,pc_Un_range)
            U,S = Symbolic_US(DataType,ground_dataset, features, features_type,N,
                            Uc,Sc, pc_U,pc_S, )
            for k in K: 
                if bad: break
                for d_flag in distances:
                    if bad: break
                    Ulabel, U_Nk = kNN_rule(k,d_flag, data, U)
                    if Ulabel == '1':
                        bad = True
                        print("Bad")
                        break
    if not bad: 
        I_.append(n)
        Uc_set.append(Uc)
        print(f"{n}: Trovato!") 
        

Uc_path = BASE_DIR / name_dataset
Uc_df_name =Uc_path/ f'Uc_{I}instances.csv'
# Salvataggio in CSV
df = ground_dataset.iloc[I_]
df.to_csv(Uc_df_name, index=False)
"""


# Uc_path = BASE_DIR / name_dataset
# Uc_df_name = f'Uc_{I}instances.csv'
# Uc_set = pd.read_excel(Uc_path, Uc_df_name) 


#
# SELEZIONARE LE 10 ISTANZE DAL DATESET IN MODO CHE RISULTANO NEGATIVE 
# PER OGNI DISTANZA, K E LIVELLO DI INCERTEZZA 

I_map = {
    'BH': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'CP': [2, 4],           
    'GC': [4, 10],              #  German Credit
    'AD': [0,1],
    'BM': [0,1],
    'BCW':[1,3],
}

I_ = I_map.get(name_dataset)

Uc_set = [dict(ground_dataset.iloc[n]) for n in I_ ]
         

Uc_df_name =f"Rule : {I_}"

#%% save info to txt
###############################################################################
# Save all experiment details in .txt
###############################################################################
params = {
# General
"seed (random)": seed,
"DataType": DataType,
# Distances
"Notion of distance": None,

# Counterfactual problem
"k": None,
"M": M,


"epsilon Huristic solver": eps_heur,

# Input U, 
"Number of inputs": I,
"U center (file)": Uc_df_name,
# Counterfactual S+x, S shape x Unknown
"S center": Sc,

# Dataset
'Dataset name': name_dataset,
"Features": features_type,
"Number of elements (N)": N,
"Number of features (J)": J,
"Uncertainty percentage range data": None,
"Uncertainty percentage Input": None,
"Uncertainty percentage counterfactual": None,


# Solver
"Exact Solver Time Limit (s)": timelim,

# Heuristic 
"Probability for GaussianVNS perturbations": beta,
"maxGaussVNS index (G)" : G,
"Local solver options" : solver_opts
}


#%%  Series of computations 
# avoid  = {"d0": [(0,0), (0,0.02)], 'd1':[(0,0)], 'dint2': [(0,0)]}
 
w =0
for pc_U in Uncert_degree:
    # percentuale incertezza dataset. Esempio: fra il 5% e il 10% ---> [0.05,0.1]
    pc_Un_range = (pc_U,pc_U)  
    params["Uncertainty percentage range data"] =  pc_Un_range
    params['Uncertainty percentage Input'] = pc_U
    for pc_S in Uncert_degree:
        params['Uncertainty percentage counterfactual'] = pc_S
    
        ###
        # Directory
        S_unc = str(int(pc_S*100))
        U_unc = str(int(pc_U*100))
        conf_name = f'U_{U_unc}_S_{S_unc}'
        conf_DIR = dataset_DIR / conf_name
        conf_DIR.mkdir(exist_ok=True)# crea le cartelle se non esistono

        #
        # Symblic data definition
        #
        data = SymbolicDataset_(DataType, ground_dataset, target, 
                                features,features_type, seed, N,pc_Un_range)
        
        for k in K:

            params['k'] = k
                
            for d_flag in distances:
                # if (pc_U, pc_S) in avoid[d_flag]:
                #     print(f"({pc_U}, {pc_S})")
                #     continue
                params['Notion of distance'] = d_flag
                # Folder
                foldername = f'{d_flag}_k{k}'
                
                # Save options
                folderpath = save_txt_(params, conf_DIR, foldername=foldername) if save else conf_DIR / foldername
                # folderpath = conf_DIR / foldername
                
                for i, Uc in zip(range(I), Uc_set):
                   
                    
                    #
                    # Symbolic definition of U and S
                    #
                    U,S = Symbolic_US(DataType,ground_dataset, features, features_type,N,
                                    Uc,Sc, pc_U,pc_S, )
                    
                    
                    #
                    # Endogenous Counterfactual computation
                    #
                    E, Nk_E, dUE, time_execution  = end_counterf_(k,d_flag, data, U)
                    results= {'input': i,'E': E.x, f'N_{k}' : list(Nk_E.keys()),  "distance type" : d_flag, "distance U-E": dUE, "Time Execution" : time_execution}
                    if save:
                        save_exel_('C-Endogenous', results, conf_DIR, foldername = folderpath)
                    
                    #
                    # Heuristic Counterfactual
                    #
                    H_ev, Nk_H_ev, d_UH_ev, t_ev, counters = Heuristic_counterf_(DataType,
                                                                                 data, features_type, features_multicat,U,S,
                                                                                 d_flag, k,M, G, J, beta, 
                                                                                 E, Nk_E,dUE, 
                                                                                 eps_heur, timelim, solver_opts)
                    

                    for h in range(0,len(H_ev)):
                        H, Nk_H, d_UH, time_exec= H_ev[h], Nk_H_ev[h], d_UH_ev[h], t_ev[h]
                        H.label = 'count'
                        U_H = {f : U.x[f] - H.x[f] for f in features }
                        results= {'H': H.x, f'N_{k}' : list(Nk_H.keys()),  "distance type" : d_flag, "distance U-H": d_UH, "U-H": U_H, "Time Execution" : time_exec}
                        print(results)
                        print('\n\n')
                        if save:
                            save_exel_(f'C-Heuristic-input{i}', results, conf_DIR, foldername = folderpath)

                # plot 
                for col_name in ['H', 'U-H']:#
                    fig, ax =  heatmap_(
                            features,
                            col_name,
                            f" k = {k}, {d_flag}, Counterfactual uncertainty : {S_unc}%, Input uncertainty: {U_unc}%",
                            folderpath )
                    fig.savefig(Path(folderpath) /f"Heatmap{col_name}.png", dpi=300, bbox_inches="tight")
#%% Heatmaps divided per input
k, d_flag =105, 'd1'
for i in range(I):
    fig, ax = heatmap_input_(
            features ,
            'H',
            f"Input {i+1} ",
            dataset_DIR,
            i ,
            Uncert_degree,
            d_flag,
            k)
    save_fold =  BASE_DIR/ "Plots in supplementary" / 'others'#name_dataset/ f"{d_flag}_k{k}"# dataset_DIR
    fig.savefig(save_fold/f"{name_dataset}_{d_flag}k{k}_Input{i}.pdf", format="pdf", bbox_inches="tight")

    