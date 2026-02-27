# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
"""
from .rectangle_class import Rettangolo
from .ball_class import ball
import numpy as np
import scipy.stats as ss
import pandas as pd
import os
from datetime import datetime
import time as tm
import json
from collections import Counter
import copy


###############################################################################
# Dataset
###############################################################################

# This function returns a dict, for each feature is associated 
# the type (categorical (also binary), integer or numerical (continuous))
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



def Dataset_selection(name):
    """
    This function select the dataset.
        -  Boston Housing
        -  German Credit
        -  Compass
    """
    
    task = 'classification'
    if name == 'BH': #boston housing
        from .BostonHousing import data
        path = "" #INSERT
        return data(path,task)
    elif name == 'GC': # german credit
        from .German import data_GC
        return data_GC()
    elif name == 'CP': # compass
        from .Compas import data_CP
        path = "" #INSERT
        return data_CP(path)
    else:
        print('To define')
    

def GroundSyntheticDataset_(seed, J, N, mean_std_0,mean_std_1):
    """
    This function generate a ground synthetic datset of J fetures called ('x1',...,"xJ")
    Half of the instances have label 1, and half 0 
    Parameters
    ----------
    seed : int
        random seed for the reproducibility
    J : int
       number of features
    N : int
        number of instances
    mean_std_0 : dict
        0-instances uniform distrubution
    mean_std_1 : dict
        0-instances uniform distrubution

    Returns
    -------
    ground_dataset : dataframe
        2 features, N elements
    target : series
        0-1 labels.
    features : list of str
        features names
    features_type : TYPE
        DESCRIPTION.

    """
    np.random.seed(seed)
    
    
    # centers and label generation
    df1 = pd.DataFrame()        # Dataframe of elements with label 1
    df0 = pd.DataFrame()        # Dataframe of elements with label 0
    df = pd.DataFrame()         # Dataframe of the dataset
    
    #Creating array of n/2  normally distributed elements x^n, 
    #mean = mean_1 , standard deviation = std_1 and label = 1
    for i in range(J):
        df1[f"x{i+1}"] = np.random.normal(loc=mean_std_1['mean'], 
                                            scale=mean_std_1['std'], 
                                            size=int(N/2))
    
    df1['Label'] = np.ones(int(N/2))
    
    #Creating array of n/2  normally distributed elements x^n, 
    #mean = mean_0 , standard deviation = std_0 and label = 0
    for i in range(J):
        df0[f"x{i+1}"] = np.random.normal(loc=mean_std_0['mean'], 
                                            scale=mean_std_0['std'], 
                                            size=int(N/2))
    
    df0['Label'] = np.zeros(int(N/2))
    
    # Merge the dataset
    df = pd.concat([df1, df0], ignore_index=True)
    print(df.shape)
    df.head()
    
    
    target = pd.Series(df['Label'])
    features =  df.columns[:-1]
    ground_dataset = df[features]
    features_type=feature_type(ground_dataset)

    return ground_dataset, target, features, features_type

def SymbolicDataset_(DataType, ground_dataset, target, features, features_type, seed, N,pc_Un_range):
    """
    pc_Un_range : intervallo percentuale di incertezza 
    calcoliamo l'incetezza -> fissata una feature f, hs[f] =  np.random.uniform(pc_l,pc_u)
    dove 
    pc_l = (1/pc_Un_range[0]) *  sum(ground_dataset[f][n] n in range(N))/N 
    pc_u = (1/pc_Un_range[1]) *  sum(ground_dataset[f][n] n in range(N))/N 
    """
    np.random.seed(seed)
    
    print(ground_dataset.shape)
    ground_dataset.head()
    
    #Bounds
    
    if DataType == 'R':
        bounds = {}
        for f in features:
            pc_l = (pc_Un_range[0]) *  (sum(abs(ground_dataset[f][n]) for n in range(N))/N) 
            pc_u = (pc_Un_range[1]) *  (sum(abs(ground_dataset[f][n]) for n in range(N))/N)
            bounds[f] =  (pc_l, pc_u) if features_type[f] == 'Numerical' else (0,0)
    elif DataType == 'B':
        J_num = sum(1 for v in features_type.values() if v == 'Numerical') # numer of numerical features
        pc_l = (pc_Un_range[0]) *  sum(sum(abs(ground_dataset[f][n]) for n in range(N) )  for  f in features if features_type[f] == 'Numerical' )/(N*J_num) 
        pc_u = (pc_Un_range[1]) *  sum(sum(abs(ground_dataset[f][n]) for n in range(N) )  for f in features if features_type[f] == 'Numerical')/(N*J_num) 
        bounds =  (pc_l, pc_u) 
    
    # Dataset with uncertainty
    data = []
    
    for n in range(N):
        x, label, name = ({f: float(ground_dataset[f][n]) for f in features },str(int(target[n])),rf"$U^{{{n+1}}}$" )
        if DataType == 'R':
            hs =  {f : np.random.uniform(bounds[f][0], bounds[f][1]) for f in features}
            U = Rettangolo(tuple(features), x, hs, label, name)
            
        if DataType == 'B':
            r =  np.random.uniform(bounds[0], bounds[1])
            U = ball(tuple(features), x, r, label, name)
            
        data.append(U)
    return data

# Symbolic Input and Counterfactual 
def Symbolic_US(DataType,ground_dataset, features, features_type,N,
                Uc,Sc, pc_U,pc_S):
    """
    Parameters
    ----------
    DataType : str --> 'R' or 'B'
    ground_dataset : dataframe
    features : list of str
    features_type : dict --{'f1':'Numerical', 'f2': categorical ..}
    N : int, number of elements of the dataset
    Uc : dict, center of the input
    Sc : dict of zeros, center of the counterfactual shape
    pc_U : percentage of input uncertainty
    pc_S : percentage of counterfactual uncertainty


    Returns
    -------
    U : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.

    """
    #
    # Input instance
    #
    if DataType =='R':
        U_hs = {} # boundaries U when R
        S_hs = {} # boundaries S when R
        for f in features:
            U_hs[f] =  pc_U * (sum(abs(ground_dataset[f][n]) for n in range(N))/N)  if features_type[f] == 'Numerical' else 0
            S_hs[f] =  pc_S * (sum(abs(ground_dataset[f][n]) for n in range(N))/N)  if features_type[f] == 'Numerical' else 0
        U = Rettangolo(tuple(features), Uc, U_hs, 'none', 'U')  #il label dovrei assicurarmi che sia zero comunque
        S = Rettangolo(tuple(features), Sc, S_hs, 'count', 'S')
    elif DataType =='B':
        J_num = sum(1 for v in features_type.values() if v == 'Numerical') # numer of numerical features
        Ur = pc_U * sum(sum(abs(ground_dataset[f][n]) for n in range(N) )  for  f in features if features_type[f] == 'Numerical' )/(N*J_num)
        Sr = pc_S * sum(sum(abs(ground_dataset[f][n]) for n in range(N) )  for  f in features if features_type[f] == 'Numerical' )/(N*J_num)
        
        U = ball(tuple(features), Uc, Ur, 'none', 'U')
        S = ball(tuple(features), Sc, Sr, 'count', 'S')
    
    
    return U, S

###############################################################################
# Distances
###############################################################################

#Eucledean distance between two points
def dE(p1,p2):
    """
    Parameters
    ----------
    p1,p2: dict

    Returns
    -------
    Eucledean distance.

    """
    features = list(p1.keys())
    return np.sqrt(sum((p1[f]-p2[f])**2 for f in features))


# min distance
def d0(U1,U2):
    """
    Parameters
    ----------
    U1,U2 : Rettangolo o ball
    
    Returns
    -------
    Their minimal distance

    """

    features = U1.features
    
    if isinstance(U1, Rettangolo):
        # a^U1_f = U1ab[f][0]     b^U1_f = U1ab[f][1]
        U1ab = U1.bounds() 
        U2ab = U2.bounds() 
        
        d_min2=  sum( max(U1ab[f][0] - U2ab[f][1] , 0, U2ab[f][0] -  U1ab[f][1])**2   for f in features)
        return np.sqrt(d_min2)
    elif isinstance(U1, ball):
         d = dE(U1.x, U2.x) - U1.r - U2.r
         return max(d,0) 
 

# max distance
def d1(U1,U2):
    """
    Parameters
    ----------
    U1,U2 : Rettangolo or ball
    
    Returns
    -------
    Their maximum distance 

    """

    features = U1.features
    
    if isinstance(U1, Rettangolo):
        # a^U1_f = U1ab[f][0]     b^U1_f = U1ab[f][1]
        U1ab = U1.bounds() 
        U2ab = U2.bounds() 
        
        d_max2= sum( max((U1ab[f][0] - U2ab[f][1])**2 ,  (U2ab[f][0] -  U1ab[f][1])**2 )   for f in features)
        return np.sqrt(d_max2)
    elif isinstance(U1, ball):
         dmax = dE(U1.x, U2.x) + U1.r + U2.r
         return dmax       

# average distance squared    
def dint2(U1,U2):
    """
    Parameters
    ----------
    U1,U2 : Rettangolo o ball
    
    Returns
    -------
    Their int2 distance 

    """
    features = U1.features
    J = len(features)
    
    if  isinstance(U1, Rettangolo):
        
        # a^U1_f = U1ab[f][0]     b^U1_f = U1ab[f][1]
        U1ab = U1.bounds() 
        U2ab = U2.bounds() 
                
        muU1 = {f : (U1ab[f][0] +U1ab[f][1] )/2 for f in features}
        muU2 = {f : (U2ab[f][0] +U2ab[f][1] )/2 for f in features}
        nu = (1/12) * (
                        sum((U1ab[f][1] -U1ab[f][0] )**2 for f in features)   
                        + sum((U2ab[f][1] -U2ab[f][0] )**2 for f in features)
                        )
    elif isinstance(U1,ball):

        muU1 = U1.x
        muU2 = U2.x
        nu = (J/(J+2))*(U1.r**2 + U2.r**2)
    
    return np.sqrt( dE(muU1, muU2)**2 + nu )


def d_label_sensitive(U1,U2, d_flag):
    """
    U2.label rule the choice of the distance. 
    """
    if d_flag == 'd0' or U2.label == '0':
        return d0(U1, U2)
    else:
        if d_flag == 'd1':
            return d1(U1, U2)
        elif d_flag == 'dint2':
            return dint2(U1, U2)
        else:
            print('Distance not recognized')
            
    
    
###############################################################################
# Calssifier Rule
###############################################################################

def kNN_rule(k,d_flag, data, U):
    """
    
    Parameters
    ----------
    k : int
        number of neighbors
    d_ : str
        flag distance in ['d0','d1','dint2']. If d='d1' or 'dint2' --> knn rule is label-sensitive
    data : list 
         Rectangles or balls
    U : Rattangolo o ball
        Inout to be classified

    Returns
    -------
    label : 0 or 1 (The classification)
    Nk : The neighborhood that classified U

    """
    Un_distances_label = {}
    for Un in data:
       d_U_Un = d_label_sensitive(U,Un, d_flag) # distance U-Un
       Un_distances_label[Un.name]   = [d_U_Un,Un.label ]
    
    # Ordina tutti per distanza
    sorted_items = sorted(
        Un_distances_label.items(),
        key=lambda item: item[1][0]
        )

    # distanza del k-esimo elemento
    kth_distance = sorted_items[k-1][1][0]
    
    # includi tutti quelli con distanza <= kth_distance
    Nk = {
        name: value
        for name, value in sorted_items
        if value[0] <= kth_distance
        }

    # estrai i label dai k elementi
    labels = [value[1] for value in Nk.values()]

    # conta le occorrenze
    counter = Counter(labels)

    # definisci il label finale
    Ulabel = '1' if counter.get('1', 0) > counter.get('0', 0) else '0'
    
    return Ulabel, Nk

###############################################################################
# Edogenous counterfactual
###############################################################################
def end_counterf_(k,d_flag, data, U):
    """
    Parameters
    ----------
    k : int
        number of neighbors.
    d_flag : str
        In ['d0','d1','dint2'] 
    data : list of Rettanglolo o ball
        ssymbolic dataset
    U : REttangolo o ball
        input

    Returns
    -------
    E : Rattangolo o ball
        Endogenous counterfactual
    Nk_E : list of neighors
    dUE : distance U from E
    te-ts : execution time
    """
    dUE = 1e8
    
    ts = tm.time()
    for Un in data:
        dU_Un = d0(U,Un) if d_flag =='d0' else d1(U,Un) if d_flag=='d1' else dint2(U,Un)
        if dU_Un < dUE: # compute neighborhood
            data_Un = data #[x for x in data if x is not Un]
            Un_kNN_label, Nk_Un = kNN_rule(k,d_flag, data_Un, Un)

            if  Un_kNN_label == '1':# and Un.label == '1':
                dUE = dU_Un
                Nk_E = Nk_Un
                E = copy.deepcopy(Un)
    te= tm.time()
                
    E.label = 'endo'
    
    return E, Nk_E, dUE, te-ts


###############################################################################
# Exact counterfactual
###############################################################################
def exact_COP_(DataType,d, d_obj, data, features, features_type, U, S,N, J,
               k, M, eps, timelim):
    from .COP import COP_
    from pyomo import environ as pym
    """
    Ricorda : dal problema ottieni
    - sk = rk**2
    - obj_value = d_**2
    - x centro del controfattuale x+S (con S centrato in Sc)
    """

    #
    # Instance of problem COP
    #
    ts =tm.time()
    instance = COP_(DataType,d, d_obj, data, features, features_type, U, S,N, J, k, M, eps)

    # 
    # Solver
    #
    solver = pym.SolverFactory('gurobi_persistent')
    solver.set_instance(instance)

    #Solver parameters
    solver.options['TimeLimit'] =timelim 


    #
    # Solve
    #
    
    solver.solve(tee = True)
    te = tm.time()
    
    #
    # Result
    #
    xC= {f: instance.x[f].value for f in features}
    Sx = copy.deepcopy(S)
    Sx.x = xC
    Sx.label = 'count'
    Nk = [i for i in range(N) if abs(instance.w[i].value) > eps]
    r_k = np.sqrt(instance.sk.value)
    d_UxS = np.sqrt(pym.value(instance.objfunction))

    #
    # Print
    #
    print("------Results------")
    print('Counterfactual center: ')
    print(xC)
    print('Neighbors indices:', *(f"U^{i + 1}" for i in Nk)) 
    print(rf"r_{{{k}}} = {r_k}")
    print(rf"Distance U-x+S ({d_obj}) = {d_UxS}")
    print(f'Exact execution time (s): {te-ts}')

    return Sx, Nk, d_UxS, te-ts
    
###############################################################################
# Heuristic support function
###############################################################################
# This function compute the variance vector sigma
def ComputeSigma_(G, J, beta, c):
    """
    

    Parameters
    ----------
    G : int
        sigma vector lenght, max gaussianVNS index
    J : int
        number of features = degree of freedom for the chisquare distribution.
    beta : float
        probability.
    c : float
        distance between heuritic initial incumbet and its nearest neighbor.

    Returns
    -------
    sigma : list
        list of G variance values.

    """
    #q: beta quantile of freedom degree J of chisqare distribution
    q = ss.chi2.ppf(beta, J)
    s = c/(np.sqrt(q))
    sigma  = [pow(2, g)*s for g in range(G)]
    return sigma  

def Heuristic_counterf_(DataType,data,features_type, features_multicat,U,S,
                        d_flag, k,M, G, J, beta, 
                        Hs, Nk_Hs, d_UHs, 
                        epsilon, timelim, solver_opts,
                        use_rel_gap : bool = False, d_UxS : float = None, rel_gap_tol: float = 1e-5):
    """
    Parameters
    ----------
    DataType : str --> 'R' or 'B'        
    data : list of Rettangolo or ball
    features_type : dict --{'f1':'Numerical', 'f2': categorical ..}
    features_multicat : dict --> {'f':['f_1', 'f_2'],..}
    U : ball or Rettangolo input
    S : ball or Rettangolo, countefactual shape
    d_flag : distance flag --> 'd0','d1','dint2'
    k : int, number of neighbors
    M : int, big M
    G : int, max GaussVNS index
    J : int, number of features
    beta : float, probability
    Hs : ball or Rettangolo, starting incumbent
    Nk_Hs : neghborhood of Hs
    d_UHs : distance U-Hs
    epsilon : float
    timelim : int, seconds, huristic timelimit
    solver_opts : Gurobi options
    use_rel_gap = False, d_UxS = None, rel_gap_tol = None : Relative gap stopping criterium

    Returns
    -------
    dict of huritic evolutions
    H_ev :  counterfactuals, balls o Rettangoli
    Nk_H_ev : neighborhoods of the counterfactuals.
    d_UH_ev : distances countefactuals H - input U
    t_ev : runtime each counterfactual has been found
    counters : Cunter --> informations about the heuristic run

    """
    #Hs, Nk_Hs, d_UHs, epsilon = copy.deepcopy(E), copy.deepcopy(Nk_E), dUE,eps_heur
    
    from .GaussVNS import GaussVNS
    
    #
    # GaussianVNS parameter
    #
    c,e = 0,0
    while c ==0 and e < k:
        Nk_Hs_1 = next( (data[i] for i in range(len(data)) if data[i].name == list(Nk_Hs.keys())[e]), None)
        c  = dE(Hs.x,Nk_Hs_1.x)   # distance between the given starting solution and element i of neighborhood Nk_HS (here i =1)
        # print(e)
        e+=1
    if c==0: c=0.1  #To be sure sigma is never 0

    sigma = ComputeSigma_(G, J,beta, c)         # GaussianVNS parameter

          
    #
    # Huristic run
    #
    H_ev, Nk_H_ev, d_UH_ev, t_ev, counters =  GaussVNS(DataType,sigma, timelim,
                                                       Hs, d_UHs, U, S,
                                                       data, features_type, features_multicat,
                                                       d_flag, k, epsilon, M, solver_opts,
                                                       use_rel_gap, d_UxS , rel_gap_tol)

    return H_ev, Nk_H_ev, d_UH_ev, t_ev, counters
###############################################################################
# Read and Save
###############################################################################
def save_txt_(params, path, foldername=None):
    """
    Salva i parametri di un esperimento in un file .txt
    
    Parameters
    ----------
    params : dict
        Dizionario contenente tutti i parametri dell'esperimento
    path : str
        Cartella in cui salvare il file
    foldername : str, optional
        Nome della cartella. Se None, viene generato automaticamente con timestamp
    """
    
    
    # Nome file automatico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"README_experiment_params_{timestamp}.txt"

    
    if foldername is None:
        folderpath = os.path.join(path, filename) 
    else:
        folderpath = os.path.join(path, foldername) 
    
    #
    # Crea la cartella
    os.makedirs(folderpath, exist_ok=True)


    filepath = os.path.join(folderpath, filename)

    with open(filepath, "w") as f:
        for key, value in params.items():
            f.write(f"{key} = {value}\n")

    print(f"Parametri salvati in: {filepath}")
    
    
    return folderpath


def save_exel_(method,results, path, foldername = None):
    """
    Salva un dizionario qualsiasi in un file Excel.
    Ogni chiamata aggiunge una riga (esperimento).
    """
    # Nome file 
    filename = f"results_{method}.xlsx"
    
    excel_path = os.path.join(path, foldername, filename) 
    
    #
    # Crea la cartella
    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)
    
    row = {}

    for key, value in results.items():

        # tipi semplici
        if isinstance(value, (int, float, str, bool)) or value is None:
            row[key] = value

        # liste, tuple, set → stringa
        elif isinstance(value, (list, tuple, set)):
            row[key] = ";".join(map(str, value))

        # dizionari → JSON
        elif isinstance(value, dict):
            row[key] = json.dumps(value)

        # fallback per oggetti arbitrari
        else:
            row[key] = str(value)

    new_df = pd.DataFrame([row])

    # append
    if os.path.exists(excel_path):
        old_df = pd.read_excel(excel_path, sheet_name='results')
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        out_df.to_excel(writer, sheet_name='results', index=False)

    print(f"Dati salvati in: {excel_path}")

    
def read_csv_to_df_(path, filename):
    file_path = os.path.join(path, filename)
    df = pd.read_csv(file_path, sep=None, engine="python")
    return df


###############################################################################
# Print
###############################################################################

# Function to print the results
def print_counterfactual(method,d_flag,k, xS, Nk, d_UxS, r_k = None):
    print("------Results------")
    if method == 'Endogenous':
        print('Endogenous Counterfactual: ')
        print(xS)
        print(f'Neighborhood (k = {k}):')
        print(Nk)
        print(rf"Distance U-x+S ({d_flag}) = {d_UxS}")
    elif method == 'Heuristic':
        print('Heuristic Counterfactual: ')
        print(xS)
        print('Neighbors indices:', *(f"U^{i + 1}" for i in Nk)) 
        print(rf"r_{{{k}}} = {r_k}")
        print(rf"Distance U-x+S ({d_flag}) = {d_UxS}")
    else:
        print('Not found.')
    
    
    
    
    
    
    
