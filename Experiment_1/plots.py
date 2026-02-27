# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 14:08:51 2026

@author: Marica Magagnini
"""
from fun import Rettangolo, ball



import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D
import os
import numpy as np


LABEL_COLOR = {
    "1": "tab:red",
    "0": "tab:blue",
    "none": "black",
    "count": "yellow",
    "c0":"skyblue",
    "c1":"pink",
    "c2":"green"   
}

###############################################################################
import math

def _points_rect_rect(A, B, d='d0'):
    """
    A e B: oggetti tipo Rettangolo, con attributi:
      - features: tuple di nomi dimensioni
      - x: dict centro
      - hs: dict semi-lati
    d = 'd0' → punti di minima distanza
    d = 'd1' → punti di massima distanza
    Ritorna (pA, pB) come dict {feature: valore}.
    """
    pA = {}
    pB = {}

    for f in A.features:
        a_c = A.x[f]
        a_h = A.hs[f]
        b_c = B.x[f]
        b_h = B.hs[f]

        a_min, a_max = a_c - a_h, a_c + a_h
        b_min, b_max = b_c - b_h, b_c + b_h

        if d == 'd0':
            # ---- MINIMA distanza lungo la dimensione f ----
            # intervalli che si sovrappongono
            if a_min <= b_max and b_min <= a_max:
                # c'è intersezione: scegliamo un punto qualunque
                lo = max(a_min, b_min)
                hi = min(a_max, b_max)
                v = 0.5 * (lo + hi)
                pA[f] = v
                pB[f] = v
            else:
                # disgiunti: prendi estremi più vicini
                if a_max < b_min:
                    pA[f] = a_max
                    pB[f] = b_min
                else:  # b_max < a_min
                    pA[f] = a_min
                    pB[f] = b_max

        elif d == 'd1':
            # ---- MASSIMA distanza lungo la dimensione f ----
            # le scelte estreme sono (a_min, b_max) oppure (a_max, b_min)
            c1_a, c1_b = a_min, b_max
            c2_a, c2_b = a_max, b_min

            if (c1_a - c1_b) ** 2 >= (c2_a - c2_b) ** 2:
                pA[f], pB[f] = c1_a, c1_b
            else:
                pA[f], pB[f] = c2_a, c2_b

        else:
            raise ValueError("d deve essere 'd0' (minima) o 'd1' (massima)")

    return pA, pB


def _points_ball_ball(A, B, d='d0'):
    """
    A e B: oggetti tipo ball, con attributi:
      - features: tuple di nomi dimensioni
      - x: dict centro
      - r: raggio
    d = 'd0' → punti di minima distanza (tra gli insiemi)
    d = 'd1' → punti di massima distanza
    Ritorna (pA, pB) come dict {feature: valore}.
    """
    # vettore tra i centri
    diff = {f: B.x[f] - A.x[f] for f in A.features}
    d_centri2 = sum(diff[f] ** 2 for f in A.features)
    d_centri = math.sqrt(d_centri2)

    pA = {}
    pB = {}

    # centri coincidenti: gestione separata
    if d_centri == 0:
        # prendiamo una direzione arbitraria (prima feature)
        first = A.features[0]
        for f in A.features:
            pA[f] = A.x[f]
            pB[f] = B.x[f]

        if d == 'd0':
            # insiemi si intersecano sempre se rA>0 e rB>0 → distanza minima = 0
            # scegliamo lo stesso punto (il centro comune)
            return pA, pB

        elif d == 'd1':
            # massima distanza = rA + rB lungo una direzione
            pA[first] -= A.r
            pB[first] += B.r
            return pA, pB

        else:
            raise ValueError("d deve essere 'd0' (minima) o 'd1' (massima)")

    # versore dalla ball A alla ball B
    u = {f: diff[f] / d_centri for f in A.features}

    if d == 'd0':
        # ---- MINIMA distanza fra gli insiemi ----
        rA = A.r
        rB = B.r

        if d_centri >= rA + rB:
            # sfere disgiunte: punti sulle superfici sulla congiungente dei centri
            for f in A.features:
                pA[f] = A.x[f] + u[f] * rA
                pB[f] = B.x[f] - u[f] * rB
        else:
            # sfere che si intersecano: distanza minima = 0
            # scegliamo un punto nella loro intersezione sulla linea dei centri
            # formula standard per la posizione del centro del cerchio di intersezione:
            # x = (d^2 + rA^2 - rB^2) / (2d) distanza da A lungo u
            rA = A.r
            rB = B.r
            x = (d_centri2 + rA**2 - rB**2) / (2 * d_centri)
            for f in A.features:
                pA[f] = A.x[f] + u[f] * x
                pB[f] = pA[f]  # stesso punto
    elif d == 'd1':
        # ---- MASSIMA distanza ----
        # sempre lungo la congiungente dei centri ma "fuori" dalle sfere
        for f in A.features:
            pA[f] = A.x[f] - u[f] * A.r
            pB[f] = B.x[f] + u[f] * B.r
    else:
        raise ValueError("d deve essere 'd0' (minima) o 'd1' (massima)")

    return pA, pB


def extreme_points(U, V, d='d0'):
    """
    U, V: due insiemi geometrici:
      - coppia di Rettangolo (con attributo 'hs') oppure
      - coppia di ball (con attributo 'r').

    d:
      - 'd0' → punti di minima distanza
      - 'd1' → punti di massima distanza

    Ritorna (pU, pV) come dict {feature: valore}.
    """
    if hasattr(U, "hs") and hasattr(V, "hs"):
        return _points_rect_rect(U, V, d=d)
    elif hasattr(U, "r") and hasattr(V, "r"):
        return _points_ball_ball(U, V, d=d)
    else:
        raise TypeError("Coppia di oggetti non riconosciuta (rettangolo-rettangolo o ball-ball).")


###############################################################################

def plotU(ax,U, *, highlight=False, name = True, linewidth=1):
    label_offset=0.05
    
    if isinstance(U, Rettangolo):
        xmin, xmax = U.bounds()['x1']
        ymin, ymax = U.bounds()['x2']
        width = 2*U.hs['x1']
        height = 2*U.hs['x2']

        color = LABEL_COLOR.get(U.label, "black")

        patch = mp.Rectangle(
            (xmin, ymin),
            width,
            height,
            fill=True,
            facecolor=color,
            edgecolor= 'black' if highlight else color,
            linewidth=linewidth
        )
        
        # posizione label
        x_label = U.x['x1']
        y_label = ymax + label_offset * height


        
    elif isinstance(U, ball):
        """
        Plotta una ball (cerchio) su ax.
        """
        color = LABEL_COLOR.get(U.label, "black")

        patch = mp.Circle(
            (U.x['x1'], U.x['x2']),     # centro
            U.r,                        # raggio
            fill=True,
            facecolor=color,
            edgecolor="black" if highlight else color,
            linewidth=linewidth
        )

        # posizione label
        x_label = U.x['x1']
        y_label = U.x['x2'] + U.r + label_offset * (2 * U.r)

    else:
        raise TypeError(f"Tipo non supportato: {type(U)}")

    ax.add_patch(patch)
    
    # --- Etichetta sopra (nera) ---
    if name:
        ax.text(
            x_label,
            y_label,
            U.name,
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            zorder=10
        )
    
def set_square_limits(ax, data, margin=0.0):
    xs, ys = [], []

    for U in data:
        if type(U) == Rettangolo:
            xmin, xmax = U.bounds()['x1']
            ymin, ymax = U.bounds()['x2']
        elif type(U) == ball:
            xmin = U.x['x1'] - U.r
            xmax = U.x['x1'] + U.r
            ymin = U.x['x2'] - U.r
            ymax = U.x['x2'] + U.r
        xs.extend([xmin, xmax])
        ys.extend([ymin, ymax])

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # centri
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    #cx,cy = (0,0)
    
    # range
    rx = xmax - xmin
    ry = ymax - ymin
    r = max(rx, ry) * (1 + margin)

    half = 0.5 * r

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)




def plot_distance(ax, p1, p2, *, linestyle="--", color="skyblue", linewidth=1.5, zorder=5):
    """
    Disegna la linea di distanza tra due punti p1 e p2.
    p1, p2: tuple (x, y)
    """
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, color=color, zorder=zorder)
    


def annotate_distance(
    ax,
    p1,
    p2,
    d,
    *,
    offset=0.15,
    fontsize=9,
    color="black"
    ):
    """
    Scrive la distanza d vicino al segmento p1-p2,
    spostata perpendicolarmente al segmento.
    """
    x1, y1 = p1
    x2, y2 = p2

    # punto medio
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)

    # vettore segmento
    vx = x2 - x1
    vy = y2 - y1
    norm = np.hypot(vx, vy)

    if norm == 0:
        return

    # vettore normale unitario
    nx = -vy / norm
    ny =  vx / norm

    # posizione testo spostata
    tx = mx + offset * nx
    ty = my + offset * ny

    ax.text(
        tx,
        ty,
        f"{d:.2f}",
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        zorder=6
    )


def plot_data(data,U,margin = float, figsize_cm=float, save : bool = False, path: str = None, filename: str = None):
    
    # cm → inch (matplotlib usa inch)
    figsize_in = figsize_cm / 2.54

    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    plotU(ax,U)             # Input
    for Un in data:
        plotU(ax,Un)

    set_square_limits(ax, data, margin=margin)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    # ax.grid(False, linestyle="--", alpha=0.4)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
    
    if save:
        # crea directory se non esiste
        os.makedirs(path, exist_ok=True)

        save_path = os.path.join(path, filename)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        print(f"Figura salvata in: {save_path}")

# calculate the pairs of points to mark the distances between S and Un in Nk 
# + the color of the distances
def compute_pSpUn_(d, S, Nk, data):
    """

    Parameters
    ----------
    d : str
        distance.
    S : Rettangolo or ball
        counterfactual
    Nk : dict 
        {Un.name : [distance, Un.label]}

    Returns
    -------
    pS_pUn : dict
        pS_pU[i] = (p1_, p2_)
    linecolor : dict
        es. linecolor[i] = 'green'

    """
    
    pS_pUn = {}
    linecolor = {}
    
    for Un_name in Nk:
        Un = next(U_ for U_ in data if U_.name == Un_name)
        if d == 'd0':
            pS_pUn[Un_name] = extreme_points(S, Un, d='d0')
            linecolor[Un_name] = 'skyblue'
        else:
            if Nk[Un_name][1] == '0':
                pS_pUn[Un_name] = extreme_points(S, Un, d='d0')
                linecolor[Un_name] = 'skyblue'
            else:
                pS_pUn[Un_name] = (S.x, Un.x) if d == 'dint2' else extreme_points(S, Un, d='d1')
                linecolor[Un_name] = 'green' if d == 'dint2' else 'pink'
    
    return pS_pUn, linecolor

# Legenda
def add_legend(ax, linecolor, DataType,counterfactual=False):
    # Distance legend (linee)
    COLOR_TO_LABEL = {
        'skyblue': r"$d_0$",
        'pink': r"$d_1$",
        'green': r"$d_{\mathrm{int}2}$",
    }
    # Marker in base al tipo di dato
    if DataType == 'R':
        marker_style = "s"   # quadratino
    elif DataType == 'B':
        marker_style = "o"   # pallino
    else:
        raise ValueError("DataType must be 'R' or 'B'")

    used_colors = []
    for c in linecolor.values():
        if c not in used_colors:
            used_colors.append(c)

    handles = [
        Line2D([0], [0],
               linestyle="--",
               linewidth=1.8,
               color=c,
               label=COLOR_TO_LABEL.get(c, "distance"))
        for c in used_colors
    ]

    # Input: quadratino nero
    handles.append(
        Line2D([0], [0],
               marker=marker_style,
               linestyle="None",
               markersize=7,
               markerfacecolor="black",
               markeredgecolor="black",
               label="Input U")
    )

    # Risultato: quadratino giallo (se disponibile)
    if counterfactual:
        handles.append(
            Line2D([0], [0],
                   marker=marker_style,
                   linestyle="None",
                   markersize=7,
                   markerfacecolor="yellow",
                   markeredgecolor="yellow",
                   label="Counterfactual")
        )

    ax.legend(handles=handles,
              loc="upper right",
              frameon=False,
              fontsize=9)

def plot_count_(d:str,U, S,data,Nk,
              margin = float, figsize_cm=float, save : bool = False,
              path: str = None, filename: str = None):
    # cm → inch (matplotlib usa inch)
    figsize_in = figsize_cm / 2.54

    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    
    
    # calculate the pairs of points to mark the distances and the color 
    pS_pUn , linecolor= compute_pSpUn_(d, S, Nk, data)

    # 1) plot rettangoli/balls (S + tutti gli U), evidenziati se stanno in N_k
    plotU(ax,U)             # Input
    plotU(ax,S,highlight = True,name = False)             # Counterfactual
    for  Un in data:
        if Un.name in Nk.keys():
            plotU(ax, Un, highlight = True, name = False)
        else:
            plotU(ax, Un, name = False)
    
      
    # 2) plot distanze minime (linee tratteggiate azzurre)

    for pS_pUn_name in pS_pUn:
        pS = tuple(pS_pUn[pS_pUn_name][0].values())
        pUn = tuple(pS_pUn[pS_pUn_name][1].values())
        plot_distance(ax, pS,pUn, linestyle="--", color=linecolor[pS_pUn_name], linewidth=1.2)

        # (opzionale) scrivi il valore della distanza a metà segmento
        # annotate_distance(
        # ax,
        # pS,
        # pUn,
        # Nk[pS_pUn_name][0], # distance
        # offset=0.2,   # regola questo valore se serve
        # fontsize=9,
        # color="black"
        # )
    
    
    
    # 3) limiti quadrati + stile

    # assi cartesiani
    # ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black", zorder=0)
    # ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black", zorder=0)


    set_square_limits(ax, [U,S] + list(data), margin=margin)
    DataType= 'R' if isinstance(U, Rettangolo) else 'B'
    add_legend(ax, linecolor,DataType, counterfactual=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.tight_layout()

    # Mostra
    plt.show()

    # Salva (meglio fare savefig PRIMA di close e usando fig.savefig)
    if save:
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        fig.savefig(save_path,format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Figura salvata in: {save_path}")
        

def plot_compare_count_(U, counterfactuals ,data,
              margin = float, figsize_cm=float, save : bool = False,
              path: str = None, filename: str = None):
    DataType= 'R' if isinstance(U, Rettangolo) else 'B'
    # cm → inch (matplotlib usa inch)
    figsize_in = figsize_cm / 2.54

    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))
    

    # 1) plot rettangoli/balls (S + tutti gli U), evidenziati se stanno in N_k
    plotU(ax,U)             # Input
    for  Un in data:
        plotU(ax, Un, name = False)
            
    all_elem_set = [U]+ list(data)
    lines_color = {}
    for i in range(len(counterfactuals)):
        Sc, d_flag, dUxS = counterfactuals.iloc[i]
        Slabel = 'c0' if d_flag == 'd0' else 'c1' if d_flag == 'd1' else 'c2'
        S = Rettangolo(U.features, Sc, U.hs,Slabel, Slabel )  if DataType== 'R' else ball(U.features, Sc, U.r, Slabel, Slabel)
        plotU(ax,S,highlight = True,name = False)             # Counterfactual
        
        # calculate the pairs of points to mark the distances and the color 
        pS_pU , linecolor= compute_pSpUn_(d_flag, S, {U.name : [dUxS, U.label]}, all_elem_set)
        pS = tuple(pS_pU[U.name][0].values())
        pU = tuple(pS_pU[U.name][1].values())
        plot_distance(ax, pS,pU, linestyle="--", color=linecolor[U.name], linewidth=1.2)
        
        all_elem_set.append(S)
        lines_color[S.name] = linecolor['U']
        
    # margin
    set_square_limits(ax, all_elem_set, margin=margin)
    add_legend(ax, lines_color,DataType, counterfactual=False)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.tight_layout()
    
    # Mostra
    plt.show()

    # Salva (meglio fare savefig PRIMA di close e usando fig.savefig)
    if save:
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        fig.savefig(save_path,format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Figura salvata in: {save_path}")