# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
"""

#
# Labraries, packages, classes
#
from pathlib import Path
import re
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_(
        features : list,
        col_name : str,
        title : str,
        directory : str ,
        pattern : str = "results_C-Heuristic-input*.xlsx",
        cmap: str = "PiYG",
        figsize : tuple = (12,6)
        ):
    
    """
   Legge tutti i file Excel con un certo pattern, estrae dall'ultima riga la colonna H
   (contenente una stringa rappresentante un dict tipo {'f1': 1.2, 'f2': 3.4, ...})
   e crea una heatmap con righe=file e colonne=keys del dict.
   
   col_name = 'H' : conterfactual values, 'U-H' : input-counterfactual difference
   """
    
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nessun file trovato in {directory} con pattern '{pattern}'")

    idx_re = re.compile(r"results_C-Heuristic-input(\d+)", re.IGNORECASE)
     
    HM_matric = pd.DataFrame(columns=features, dtype=float)
    for f in files:
        m = idx_re.search(f.stem)
        if not m:
            continue

        idx = int(m.group(1))

        df = pd.read_excel(f)
        
        #
        if df.shape[0] == 0:
            raise ValueError("ERROR: Empty file.")
            
        Hc = ast.literal_eval(df[col_name].iloc[-1])
        HM_matric.loc[idx+1] = Hc
    

    # Heatmap
    cmap = "PiYG" if col_name == 'U-H' else "YlGnBu"
    

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        HM_matric,
        cmap=cmap,
        vmin=0 if col_name == 'H' else -0.3,
        vmax=1 if col_name == 'H' else 0.3,
        linewidths=0.3,
        linecolor="white",
        cbar=True
    )
    ax.set_title( title)
    ax.set_xlabel("Features")
    ax.set_ylabel("Input instances")
    


    plt.tight_layout
    return fig, ax


def heatmap_input_(
        features : list,
        col_name : str,
        title : str,
        BASE_dir : str ,
        i : int,
        Uncert_degree : list,
        d_flag: str,
        k : int,
        pattern : str = "results_C-Heuristic-input",
        cmap: str = "PiYG",
        figsize : tuple = (12,6)
        ):
    
    filename = pattern + str(i) + ".xlsx"
    
    HM_matric = pd.DataFrame(columns=features, dtype=float)
    for pc_U in Uncert_degree:
        for pc_S in Uncert_degree:

            ###
            # Directory
            S_unc = str(int(pc_S*100))
            U_unc = str(int(pc_U*100))
            conf_name = f'U_{U_unc}_S_{S_unc}'
            conf_DIR = BASE_dir / conf_name / f'{d_flag}_k{k}'
            
            df = pd.read_excel(conf_DIR / filename)
            
            #
            if df.shape[0] == 0:
                raise ValueError("ERROR: Empty file.")
                
            Hc = ast.literal_eval(df[col_name].iloc[-1])
            HM_matric.loc[f"U {U_unc}%, S {S_unc}%"] = Hc
            
    # Heatmap
    cmap = "PiYG" if col_name == 'U-H' else "YlGnBu"
    

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        HM_matric,
        cmap=cmap,
        vmin=0 if col_name == 'H' else -0.3,
        vmax=1 if col_name == 'H' else 0.3,
        linewidths=0.3,
        linecolor="white",
        cbar=True
    )
    ax.set_title( title)
    ax.set_xlabel("Features")
    ax.set_ylabel("Counterfactuals")
    


    plt.tight_layout
    
    return fig, ax