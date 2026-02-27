# -*- coding: utf-8 -*-
"""
@author: Marica 

Experiment 1

"""

#
# Labraries, packages, classes
#
from fun import GroundSyntheticDataset_ , SymbolicDataset_,Symbolic_US
from fun import d_label_sensitive,  kNN_rule, exact_COP_, end_counterf_,Heuristic_counterf_
from fun import save_txt_, save_exel_
import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

User = 'Maric' #'Maric' 'magag'

def rel_gap(d_ex, d_heur):
    return (d_heur - d_ex)/abs(d_ex)


def plot_gapcurves_inputs_xlsx(
    directory,
    pattern="GapCurve_input*.xlsx",
    time_col="time",
    gap_col="gap",
    show=False,
):
    """
    Plotta tutte le GapCurve_input{i}.xlsx nella directory.
    - gap < 0 viene forzato a 0
    - asse x: 0 -> max time globale
    - asse y: 0 -> max gap globale
    - legenda: i (1..10)
    """

    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nessun file trovato in {directory} con pattern '{pattern}'")

    idx_re = re.compile(r"GapCurve_input(\d+)$", re.IGNORECASE)

    curves = []
    max_time = float("-inf")
    max_gap = float("-inf")

    for f in files:
        m = idx_re.search(f.stem)
        if not m:
            continue

        idx = int(m.group(1))

        df = pd.read_excel(f)
        if time_col not in df.columns or gap_col not in df.columns:
            raise ValueError(
                f"{f.name}: colonne richieste non trovate. "
                f"Attese: '{time_col}', '{gap_col}'. "
                f"Trovate: {list(df.columns)}"
            )

        d = df[[time_col, gap_col]].copy()
        d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
        d[gap_col] = pd.to_numeric(d[gap_col], errors="coerce")
        d = d.dropna()

        # 🔒 clamp: gap negativo → 0
        d[gap_col] = d[gap_col].clip(lower=0)

        d = d.sort_values(time_col)

        if d.empty:
            continue
        
        mask = d["gap"] < 0.001
        t_first = d.loc[mask, "time"].iloc[0] if mask.any() else float(d[time_col].max())
        max_time = max(max_time, t_first)
        max_gap = max(max_gap, float(d[gap_col].max()))

        curves.append((idx, d))
    
    # max_time = 60
    # max_gap = 0.3
    max_time += max_time/10
    max_gap += max_gap/10
    
    for cu in curves:
        cu_end_gap = list(cu[1][gap_col])[-1]
        cu[1].loc[len(cu[1])]  = {time_col : max_time, gap_col : cu_end_gap}
    
    if not curves:
        raise ValueError("Nessuna curva valida trovata.")

    curves.sort(key=lambda x: x[0])  # legenda ordinata

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, d in curves:
        ax.plot(d[time_col].values, d[gap_col].values, label=str(idx+1))

    ax.set_xlim(0, max_time if max_time > 0 else 1)
    ax.set_ylim(0, max_gap if max_gap > 0 else 1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Gap")
    ax.set_title("Relative Gap vs Time heuristic computation")
    ax.legend(title="Input", loc="best")
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()

    return fig, ax



#%% Fixed Parameters

# Folder 
save = False

ConfName_ = ['Configuration_1','Configuration_2'] #, 'Configuration_3']
# path  Experiment_1
BASE_DIR = Path(__file__).resolve().parent

name_dataset = 'Synthetic'                      
DataTypes = ["R","B"]                           #"R" rectangles or "B" balls

distances =['d0','d1','dint2']                  # 'd0' : all min dist, 'd1' or 'dint2': max  or int2 dist for label 1

# Counterfactual problem
K = [3,5,7]                                     # Elements in the neigborhood
M =  2*1e3                                      # Big-M 
alpha = 1e-1*5                                  # Probability of the classifier
eps_exact = 1e-10                               # Nk construction
eps_heur = 1e-5                                 # Local costraints

#Dataset
N = 30                                          # Number of elements in the dataset
J = 2                                           # Number of features
Sc = {'x1':0, 'x2':0}                           # Counterafactual shape center
features_multicat = {}

#{name: [seed, mean_0, mean_1, var_0, var_1, pc_Un_range, pc_U,pc_S]}
configurations = {
    'Configuration_1': [10, 2, 10, 64,64, (0.04,0.09),0.04,0.04] ,
    'Configuration_2': [23, 4, -2, 30,30, (0.04,0.09),0.06, 0.06] ,
    'Configuration_3': [23, 4, -2, 30,30, (0.04,0.09),0.06, 0]  
    }

# (x1l,x1u), (x2l,x2u)
conf_INPUTS_ = {'Configuration_1':  (-10, 8,-5, 8),
'Configuration_2': (5,15,0,10)  ,
'Configuration_3': (5,15,0,10) } 
I = 10

    
###############################################################################
# Solver-Heuristic params
###############################################################################
timelim = 300                               # Exact and Heuristic time limit

beta = 0.99                                 # Probability for GaussianVNS perturbations
G = 6                                      # number of sigma components, maxGaussVNS index

solver_opts = {'TimeLimit': 1}

###############################################################################
# Save all experiment details in .txt
###############################################################################
params = {
# General
"seed (random)": None,
"DataType": None,
# Distances
"Notion of distance": None,

# Counterfactual problem
"k": None,
"M": M,
"alpha": alpha,
"epsilon exact solver": eps_exact,
"epsilon Huristic solver": eps_heur,

# Input U, 
"Number of inputs": I,
"U center": None,
# Counterfactual S+x, S shape x Unknown
"S center": Sc,

# Dataset
'Dataset name': name_dataset,
"Features": ['x1', 'x2'],
"Number of elements (N)": N,
"Number of features (J)": J,
"Uncertainty percentage range data": None,
"Uncertainty percentage Input": None,
"Uncertainty percentage counterfactual": None,
"0-label distribution": None,
"1-label distribution": None,

# Solver
"Exact Solver Time Limit (s)": timelim,

# Heuristic 
"Probability for GaussianVNS perturbations": beta,
"maxGaussVNS index (G)" : G,
"Local solver options" : solver_opts
}
 

#%%  Series of computations 
w =0
for co_ in ConfName_:
    
    ###
    # Directory
    conf_DIR = BASE_DIR / co_
    conf_DIR.mkdir(exist_ok=True)# crea le cartelle se non esistono
    
    # Synthetic Dataset configuration
    co_item = configurations[co_]
    seed = co_item[0]                               # Random generation seed
    mean_std_0 = {'mean' : co_item[1], 'std' : np.sqrt(co_item[3])}
    mean_std_1 = {'mean' : co_item[2], 'std' : np.sqrt(co_item[4])}
    
    ground_dataset, target, features, features_type = GroundSyntheticDataset_(seed, J, N, mean_std_0,mean_std_1)
    # save_exel_('dataset',ground_dataset.to_dict(orient="index"), conf_DIR, foldername = '')
    
    # percentuale incertezza dataset. Esempio: fra il 5% e il 10% ---> [0.05,0.1]
    pc_Un_range = co_item[5]   
    pc_U,pc_S = co_item[6], co_item[7]    # percentuale incertezza input e controfattuale
    
    # Update params
    params["seed (random)"], params["Uncertainty percentage range data"] = seed,  pc_Un_range
    params['Uncertainty percentage Input'], params['Uncertainty percentage counterfactual'] = pc_U,pc_S
    params['0-label distribution'], params['1-label distribution'] =mean_std_0, mean_std_1 

    # CREA LA LISTA DI INPUT
    x1l,x1u, x2l,x2u =conf_INPUTS_[co_]
    INPUTS_ = { i : {'x1': np.random.uniform(x1l,x1u)  , 'x2': np.random.uniform(x2l,x2u)} for i in range(I)}
    # save_exel_('inputs',INPUTS_, conf_DIR, foldername = '')
    params['U center'] = INPUTS_
    ###
    
    for DataType in DataTypes:
        path = conf_DIR / f"{DataType}"
        path.mkdir(exist_ok=True)        
        params['DataType'] = DataType
        
        #
        # Symblic data definition
        #
        data = SymbolicDataset_(DataType, ground_dataset, target, features,features_type, seed, N,pc_Un_range)
        
        for k in K:

            params['k'] = k
                
            for d_flag in distances:
                params['Notion of distance'] = d_flag
                # Folder
                foldername = f'{d_flag}_k{k}'
                
                # Save options
                folderpath = save_txt_(params, path, foldername=foldername) if save else path / foldername

                
                for i in  range(10): #Input centerI
                    
                    if True: #k in re_do[d_flag] and i in re_do[d_flag][k]:
                
                        w +=1
                        
                        # Define Input i
                        Uc = INPUTS_[i]                   
                        U,S = Symbolic_US(DataType,ground_dataset, features, features_type,N,
                                        Uc,Sc, pc_U,pc_S, )
                        
                        #
                        # Exact Counterfactual computation
                        #
                        d_obj = d_flag
                        Sx, Nk_index, d_UxS,time_exact = exact_COP_(DataType,d_flag, d_obj,
                                                                    data, features, features_type, U, S,N, J,
                                                                    k, M, eps_exact, timelim)
                        Nk_Sx = {}
                        for n in range(N):
                            if n in Nk_index:
                                Nk_Sx[data[n].name] = [d_label_sensitive(Sx,data[n], d_flag), data[n].label]
                        
                        results= {'input': i, 'xS': Sx.x, f'N_{k}' : list(Nk_Sx.keys()),  "distance type" : d_flag, "distance U-x+S": d_UxS, "Time Execution" : time_exact}
                        if save:
                            save_exel_('C-Exact', results, path, foldername = folderpath)
                            
                        #
                        # Endogenous Counterfactual computation
                        #
                        E, Nk_E, dUE, time_execution  = end_counterf_(k,d_flag, data, U)
                        results= {'input': i,'E': E.x, f'N_{k}' : list(Nk_E.keys()),  "distance type" : d_flag, "distance U-E": dUE, "Time Execution" : time_execution}
                        if save:
                            save_exel_('C-Endogenous', results, path, foldername = folderpath)
                        
                        #
                        # Heuristic Counterfactual
                        #
                        H_ev, Nk_H_ev, d_UH_ev, t_ev, counters = Heuristic_counterf_(DataType,
                                                                                     data, features_type, features_multicat,U,S,
                                                                                     d_flag, k,M, G, J, beta, 
                                                                                     E, Nk_E,dUE, 
                                                                                     eps_heur, timelim, solver_opts,
                                                                                     True,  d_UxS)
                        
                        GapCurve = {}
                        for h in range(0,len(H_ev)):
                            H, Nk_H, d_UH, time_exec= H_ev[h], Nk_H_ev[h], d_UH_ev[h], t_ev[h]
                            H.label = 'count'
                            results= {'H': H.x, f'N_{k}' : list(Nk_H.keys()),  "distance type" : d_flag, "distance U-H": d_UH, "RelGap": rel_gap(d_UxS, d_UH), "Time Execution" : time_exec}
                            GapCurve[h] = {'time' : time_exec,'gap': rel_gap(d_UxS, d_UH)}
                            # print(GapCurve[h])
                            if save:
                                save_exel_(f'C-Heuristic-input{i}', results, path, foldername = folderpath)
                                df_gap = pd.DataFrame.from_dict(GapCurve, orient="index") 
                                df_gap.to_excel(Path(folderpath) / f"GapCurve_input{i}.xlsx", index=False)
                         
                # plot 
                fig, ax = plot_gapcurves_inputs_xlsx(folderpath)
                filename_ =  f"GapCurves_all_inputs_{DataType}_{d_flag}k{k}"
                save_fold = Path(folderpath)
                fig.savefig( save_fold / filename_, format="pdf", bbox_inches="tight")
            
                
                        
#%%  Series of computations 

"""
# Check input instance if 0-label
Ulabel, U_Nk = kNN_rule(k,d_flag, data, U)
if Ulabel == '1':
    print(f'{co_}, DataType {DataType}, k = {k}, {d_flag}')
    print(f'U label for sensitive distance kNN rule: {Ulabel}, \n Neighbors: {U_Nk}')

"""
     
     
