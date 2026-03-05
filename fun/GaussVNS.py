# -*- coding: utf-8 -*-
"""


@author: Marica Magagnini
This is the implemantation of the GaussianVNS heuristic. 


"""
import numpy as np
import copy
from .functions import kNN_rule, d_label_sensitive
import time

from pyomo import environ as pym
from pyomo.opt import TerminationCondition, SolverStatus
from .local_COP import local_COP_

from collections import Counter

seed = None
rng = np.random.default_rng(seed)


def not_stop_(t,timelim, use_rel_gap : bool , RelGap : float , rel_gap_tol: float  ) -> bool:
    """
    Continue Gauss-VNS while criterium based on timelimit 
    or timelimit + relativeGap (if exact sol is exailable)    

    Parameters
    ----------
    t : current time passed
    timelim : time limit overall while execution
    use_rel_gap : bool
        Use Relative gap as second stop criterium
    RelGap : float
        current realative gap
    rel_gap_tol : float
        relative gap tolerece.  Stop when RelGap < rel_gap_tol

    Returns
    -------
    bool
        True is stop criterium has not been reach yet

    """
    if use_rel_gap:
        return (t < timelim) and (RelGap > rel_gap_tol)
    else:   
        return (t < timelim)


def rel_gap(d_exact, d_heur):
    """
    # Relative Gap function computation
    d_exact =  distance U to exact counterfactual 
    d_heur = distance U to incumbent heuristic counterfactual 
    """
    return (d_heur - d_exact)/abs(d_exact)


def update_g_(g, G):
    # This function update the GaussVNS Neighborhood g in [0, .., G-1], to each g
    # is associated a spesific variance value.
    if g < G-1:
        return g+1
    else:
        return 0
    
 
def dinamic_G_(g, Gmax, timelim, t):
    """
    Increment the G-VNS index as the time passing. 
    G is a integer upper bound for the incrementation of index g
    Parameters
    ----------
    g : int, current index
    Gmax : int, user-input max G
    timelim : time limit
    t : current time passed
    Returns
    -------
    G : int, g upper bound
    """
    time_step_range = timelim/Gmax
    step = int(np.floor(t/time_step_range))
    G = int(Gmax/2) + step if int(Gmax/2) + step < Gmax else Gmax
    return G

###############################################################################
def perturbation_(sg, H, features_type, features_multicat):
    """
    This function pertubs the curren incumbent H to H_star. 
    Perturbation rules:
        - if feature is categorical: random flip coin
        - if fature is numericaal: H_star.x  = H.x + sigma*tau, sigma variance 
        and tau from a normal distribution N(0,1).
    Parameters
    ----------
    sg : float
        variance
    H : Rettangolo or ball
        Heuristic incumbent
    features_type : dict
        features associated to their time
    features_multicat : dict
        multicat features' groups

    Returns
    -------
    H_star : Rettangolo or ball
        Perturbed incumbent

    """

    multicat_set = {f for group in features_multicat.values() for f in group}

    tau = {}

    # tau for no-multicat fetures
    for f, ftype in features_type.items():
        if f in multicat_set:
            continue
        tau[f] = rng.integers(0, 2) if ftype == "Categorical" else rng.normal()
        # if categorical binary case: coin flip (if 1: feature changes else not)
        # else continuous case ( Gaussian distribution mean 0, std 1)

    #  tau for multicat feature :  1 per group
    for group in features_multicat.values():
        old_f_1 =  next((f for f in group if H.x[f] == 1), None)
        new_f_1 = rng.choice(group)
        tau.update({f: 1 if f == old_f_1 or  f == new_f_1 else 0 for f in group})


    # H_star.x  = H.x + sigma*tau if numerical, H_star.x  = (H.x + tau)%2 if categorical
    H_star = copy.deepcopy(H)
    H_star.name = 'perturbation'
    H_star.label = None
    H_star_center = {}
    for f, ftype in features_type.items():
        if ftype == 'Categorical':
            H_star_center[f] = int(H.x[f] + tau[f]) %2
        else:
            H_star_center[f] = H.x[f] + sg*tau[f]
    H_star.x  = H_star_center
    return H_star

def Nk_IIlayer(k,d_flag, data, U):
    """
    This function compute the II-layer neighbors of U. 
    Sorting the neighbors from the closest to the furthest, select the k-neighbors 
    with 0-label from k+1^th distances. Return also the next 0-label element of 
    the series.
    
    Parameters
    ----------
    k : int
        number of neighbors
    d_flag : str
        flag distance in ['d0','d1','dint2']. If d='d1' or 'dint2' --> knn rule is label-sensitive
    data : list 
         Rectangles or balls

    U : Rattangolo o ball
        reference instance 

    Returns
    -------
    Nk_II : dict {Un.name: [d_UUn, Un.label]}, negative II-layer neighborhood 
    N0_III : ball or Rettangolo,  the following 0-label after the II-layer
    """
    Un_distances_label = {}
    for Un in data:
       d_U_Un = d_label_sensitive(U,Un, d_flag) # distance U-Un
       Un_distances_label[Un.name]   = [d_U_Un,Un.label ]
    
    # Sort, increasing distances
    sorted_items = sorted(
        Un_distances_label.items(),
        key=lambda item: item[1][0]
        )

    # k-th element distance
    kth_distance = sorted_items[k-1][1][0]
    
    # folter "II-layer" element: distance > kth_distance and label == '0'
    candidates = [
        (name, value)
        for name, value in sorted_items
        if value[0] > kth_distance and value[1] == '0'
    ]

    # select first k candidates
    Nk_II = dict(candidates[:k])
    N0_III =  next( (copy.deepcopy(Un) for Un in data if Un.name == candidates[k][0] )) 
    return  Nk_II, N0_III

# Compare if Nk ==  Mk, Nk, Mk neighborhoods
def equal_neighs(Nk, Mk):
    return Nk.keys() == Mk.keys()   


def perturbation_eval(g,G, counters, Nk_Hstar, H_star_label,Nk_H ) -> float:
    """
    This function evaluates if the perturbation is feasible to solve a local problem. 
    -- > Nk_Hstar must be composed of majority label-1 elements -->H_star.label='1'
    solve_local : bool
        True: proceed to local problem, False:forbid to solve local problem .
    g : updated parameter

    """
    solve_local =  False
    # H_star not counterfactual
    if H_star_label == '0':
        counters['Nk_star_0'] += 1                  # Increment bad neighboorhood counter
        g = update_g_(g,G)
        print("The neighborhood of the perturbation cannot yield a counterfactual.")
        
    #Equal neighborhoods
    elif equal_neighs(Nk_H, Nk_Hstar):
        counters['same'] +=1                        # Increment same Neighborhoods counter
        g = update_g_(g,G)
        print("The neighborhood of the perturbation is equal to the current incumbent.")                   
    else:
        solve_local =  True
    return solve_local, g

###############################################################################


def ro_(d_flag, Lambda, k, H_inc, Nk_Hinc, N_ref,data):
    """
    # reserch region radius computation 
    Parameters
    ----------
    d_flag : str, 'd0', 'd1' or 'dint'
    Lambda : float, reserch region magnitude
    k : int, number of neighbors
    H_inc : ball or Rettangolo, current incumbent
    Nk_Hinc: dict, incumbent neighborhood
    N_ref : ball or Rettangolo, reference element
    data : list, dataset

    Returns
    -------
    float 
        radius ro
    """
    
    # i-th neighbor of the current incumbent (if -1, the kth)
    i = -1 #  i+1-th neighbor
    Nk_Hinc_i = next( (Un for Un in data if Un.name == list(Nk_Hinc.keys())[i]), None) #-1
    
    dist_Ni_Hinc = d_label_sensitive(H_inc, Nk_Hinc_i, d_flag) # distance Hinc to its i-th neighbor
    dist_Nref_Hinc = d_label_sensitive(H_inc, N_ref, d_flag) # distance Hinc to N0_III
    Hinc_weight= d_label_sensitive(H_inc, H_inc, d_flag)
    """
    Hinc_weight is 0 when d_flag is 'd0', and > 0 otherwise. 
    """
    return Hinc_weight + Lambda * (dist_Nref_Hinc - dist_Ni_Hinc)


# Retrive a solution from local instance computation 
def sol_(instance, features, S):
    xC= {f: instance.x[f].value for f in features}
    Hcop = copy.deepcopy(S)
    Hcop.x, Hcop.label, Hcop.name = xC, None, 'x+S'
    d_UHcop = np.sqrt(pym.value(instance.objfunction))
    return Hcop, d_UHcop 

# Solve a local problem instance
def Solve_LocalProblem_(DataType, d, d_obj, k , data, features_type, features_multicat, U,S,Hl,
                        Nk_IL, Nk_IIL_0, ro, epsilon,M, counters, solver_opts):
    
    """
    Solve an instance of the local problem for fixed
    - radius ro
    - incumbent Hl an its neighborhood Nk_IL
    - II-layer neigborhood Nk_IIL_0
    
    Retrive, if available acounterfactual Hcop, its neighborhood Nk_Hcop and 
    the obj fuction value d_UHcop
    """
    
    # Instance
    instance  = local_COP_(DataType,d,d_obj, 
                           data, features_type,features_multicat, U, S, Hl, 
                           Nk_IL, Nk_IIL_0, ro, epsilon, M)
    
    # Solver
    opt = pym.SolverFactory('gurobi_persistent')
    opt.set_instance(instance)
    
    # Solve
    ts = time.time()
    try:
        results = opt.solve(save_results=False,tee=True,options=solver_opts) 
        grb = opt._solver_model   # oggetto gurobipy.Model
        # --- leggi il numero di soluzioni nella pool Gurobi ---
        solcount = int(grb.SolCount)
        print("Gurobi pool size (SolCount):", solcount)
        
        solver_status  = results.solver.status 
        term_cond = results.solver.termination_condition
        
        print(f'Numero di soluzione trovate: {solcount}')
    except Exception as e:
        solver_status, term_cond, grb, solcount= None, None, None, 0
        print(f"[ERROR] Errore durante la risoluzione con Gurobi: {e}")    
    te = time.time()
    print(f'Local problem Execution time : {te-ts} s')
           
    Hcop, Nk_Hcop, d_UHcop = None, None, None  
    
    #
    # Solver Termination Condition
    #
    print(f'Solver Status: {solver_status}, Termination Condition : {term_cond}' )
    if solcount > 0: # CONTROLLA CHE  Gurobi abbia trovato un'altra soluzione oltre quella in input
        
        counters['cop_ok'] +=1
       
        # retrive solution
        Hcop, d_UHcop = sol_(instance, list(features_type.keys()), S)
        
        # Compute its neighborhood
        Hcop.label, Nk_Hcop = kNN_rule(k,d, data, Hcop)
        
       
    else :
        # local_solution = False
        counters['cop_aborted'] +=1
        # Hcop, Nk_Hcop, d_UHcop = copy.deepcopy(H_star), copy.deepcopy(Nk_IL), d_label_sensitive(U, H_star, d)
        # Hcop.label, Hcop.name = 'count', 'perturb'
        
    return   Hcop, Nk_Hcop, d_UHcop



    
   
    
    

def Local_search_(counters, d_flag,k, DataType,
                  data,features_type, features_multicat, U, S,
                  H_star, Nk_Hstar,
                  epsilon, M, solver_opts, refinement):
    """
    Solve multiple instance of the local problem updating reserch region till 
    the retrive solution is still a counterfactual.

    Parameters
    ----------
    counters : variable Counter
    k : int
        number of neighbors
    d_flag : str
        flag distance in ['d0','d1','dint2'].
    DataType : 'R' or 'B'
    data : list 
         Rectangles or ball
    features_type : dict
        {feature name : Numerical or categorical}.
    features_multicat : dict
        {feature name : [feature_name_1, feature_name_2..]}
    U : Rattangolo o ball
        input  instance 
    S : Rattangolo o ball
        counterfactual shape.
    H_star : Rattangolo o ball
        perturbation.
    Nk_Hstar : dict
        perturbation neighborhood.
    epsilon : float
    M : int, big M
    solver_opts : dict
    refinement: bool
        If True,Nk_Hl remains fixed and local problem is solve for elarging lambda 
            (the radius of the research region ro i a multiple of lambda)
        else, Nk_Hl can be updated but lambda is fixed to 1.

    Returns
    -------
    Hl : Rattangolo o ball
        best countefactual from the multiple local instances resolutions
    Nk_Hl_IL : dict
                Hl neighborhood
    d_UHl : float
            distance Hl-U
            """
    
    
    #Lambda : research region magnitude
    Lambda = 1e-5 if refinement else 1
    search  = True  
    Hl, Nk_Hl_IL, d_UHl = copy.deepcopy(H_star), copy.deepcopy(Nk_Hstar) ,d_label_sensitive(U, H_star, d_flag)
    Nk_Hl_IIL_0, ref_= None, None  
      
    while search:
        counters['refinement'] +=1
        
        #
        # 0-label II-layer of Hl and the ref_ element 
        #
        Nk_Hl_IIL_0_new = Nk_IIlayer(k,d_flag, data, Hl)[0]
        # k+1-th neighbord if refinement else 2k^th 0-label  
        ref_name = list(kNN_rule(k+1,d_flag, data, Hl)[1])[-1]  if refinement else list(Nk_Hl_IIL_0_new)[-1]
        ref_new =  next( (Un for Un in data if Un.name == ref_name), None)
        
        #
        # Increment Lambda if N0_Hl_IIIL_new == N0_Hl_IIIL and Nk_Hl_IIL_0 == Nk_Hl_IIL_0_new
        #
        if ref_ == ref_new  and Nk_Hl_IIL_0.keys() == Nk_Hl_IIL_0_new.keys():
            if refinement:
                counters['lambda_inc']+=1
                # Lambda += abs(np.log(Lambda/1000))/1000  # Research region magnitude
                Lambda += 1e-5
            else:
                search  = False
                break
            
        else:
            ref_ = copy.deepcopy(ref_new)
            Nk_Hl_IIL_0 = copy.deepcopy(Nk_Hl_IIL_0_new)
       
        #
        # research region radius             
        #
        ro = ro_(d_flag, Lambda,k, Hl, Nk_Hl_IL,  ref_,  data)                   
        
        #
        # Local problem  instance 
        #
        Hcop, Nk_Hcop, d_UHcop = Solve_LocalProblem_(DataType, 
                                                     d_flag, d_flag,
                                                     k , data,
                                                     features_type, features_multicat, 
                                                     U,S,Hl, Nk_Hl_IL, Nk_Hl_IIL_0, 
                                                     ro, epsilon, M, 
                                                     counters, solver_opts)
        #
        # Local solution evaluation
        #
        if Hcop and Hcop.label == '1' and d_UHl - d_UHcop  > 1e-15:
            
            Hcop.label, Hcop.name = 'count', 'locCOP'
            Hl, Nk_Hl_IL, d_UHl = copy.deepcopy(Hcop), copy.deepcopy(Nk_Hcop), d_UHcop
            counters['cop_count'] +=1
         
        else:
            search = False                      # Stop refining solution
            counters['cop_notCount'] +=1
            Hl.label, Hl.name = 'count', 'perturb'

    
    return Hl, Nk_Hl_IL, d_UHl



###############################################################################
def print_counters(counters: Counter) -> None:
    descriptions = {
        'w': 'Number of while iterations',
        'same': 'Number of same neighbors after perturbation (Nk_H* = Nk_H)',
        'Nk_star_0': 'Number of Nk_H* with 0-labels majority',
        'solve_cop': 'Number of local problem instance solved',
        'cop_aborted': 'Number of local problem instances solved without retrieving a solution',
        'cop_ok': 'Number of local problem instances  solved retrieving at least one solution',
        'cop_count': 'Number of times the local problem yields a counterfactual',
        'cop_notCount': 'Number of times the local problem did not yield a counterfactual',
        'better': 'Number of new better incumbents',
        'worse': 'Number of worse incumbents',
        'lambda_inc': 'Number of times Lambda has been incremented'
    }
    
    print("Counters summary:")
    print("-" * 60)
    for key, desc in descriptions.items():
        value = counters.get(key, 0)
        print(f"{key:15s} : {value:8d}  | {desc}")
    print("-" * 60)
    
    
def print_evolution(H_ev, Nk_H_ev, d_UH_ev, t_ev, verbose=False):
   """
   Print the evolution of incumbents.
   
   Parameters
   ----------
   H_ev : dict
       Evolution of H
   Nk_H_ev : dict
       Evolution of Nk_H
   d_UH_ev : dict
       Evolution of d_UH
   t_ev : dict
       Time when the incumbent was found
   verbose : bool
       If True, prints full H and Nk_H content
   """
   
   print("\nEvolution of incumbents")
   print("=" * 70)
   header = f"{'#':>4} | {'time (s)':>10} | {'d_UH':>10}"
   
   if not verbose:
       print(header)
       print("-" * len(header))
   
   for h in sorted(H_ev.keys()):
        Hx = H_ev[h].x
        Nk_H = Nk_H_ev[h]
        d_UH = d_UH_ev[h]
        t = t_ev[h]
        
        if verbose:
            print(header)
            print("-" * len(header))
            print(f"{h:4d} | {t:10.3f} | {d_UH:10.4f} ")
            
            print("\n   H features     :")
            for f in Hx:
                print(f'{f} : {Hx[f]}')
            print("\n   Nk_H  :")
            for ng in Nk_H:
                print(f'{ng}: {Nk_H[ng]}')
            print("-" * 70)
        else:
            print(f"{h:4d} | {t:10.3f} | {d_UH:10.4f} ")
            
   print("=" * 70)
   
###############################################################################

def GaussVNS(DataType,sigma, timelim,
             Hs, d_UHs, U, S, data, features_type, features_multicat,
             d, k, epsilon, M, solver_opts,
             use_rel_gap : bool, d_UxS : float, rel_gap_tol: float):
    
    # Parameters
    G_input= len(sigma)
    refinement = False
    ###########################################################################
    # Initialization
    ###########################################################################
    g = 0                                                                           # GaussVNS index
    
    # Incumbent
    H = copy.deepcopy(S) 
    H.x = Hs.x
    H.label , Nk_H = kNN_rule(k,d, data, H)
    d_UH = copy.deepcopy(d_UHs)   
    
    t_start, t, t_newH_found = time.time(), 0, 0                                    # Time
    RelGap = rel_gap(d_UxS, d_UH) if use_rel_gap else None
    ###########################################################################
    # Follow the huristic evolution
    ###########################################################################
    counters = Counter()  # See desctiption in print_counters

    H_ev, Nk_H_ev, d_UH_ev, t_ev = {},{},{},{}
    h =0
    H_ev[h], Nk_H_ev[h], d_UH_ev[h], t_ev[h] = H, Nk_H, d_UH, 0 # starting incumbent
    
    ###########################################################################
    # GaussVNS CYCLE
    ###########################################################################
    #while (g < G) and (t < timelim) :
    while  not_stop_(t,timelim, use_rel_gap,  RelGap, rel_gap_tol ):
        counters['w'] +=1                   # increment while iteration
        # G = dinamic_G_(g, G_input, timelim, t) # Dynamic index
        G = G_input
        
        # First iteration directly solve local problem for the initial solution
        # with desidered shape S (If Hs with shape S is a counterfactual, else perturbation)
        if counters['w'] == 1 and H.label == '1':
            H_star, Nk_Hstar_IL = copy.deepcopy(H), copy.deepcopy(Nk_H)
            go_To_localProblem =  True
            
        else: 
            #
            # Perturbation
            #
            H_star = perturbation_(sigma[g], H, features_type, features_multicat)   # H*
            H_star_label, Nk_Hstar_IL = kNN_rule(k,d, data, H_star)                 # Nk(H*) : I-layer neighbors of H* (k-nearest neighbors)
     
            #
            # Evaluation of the perturbation
            #
            # If H_star counterfactual, got to solve local problem else increment g
            go_To_localProblem, g = perturbation_eval(g,G, counters, 
                                                      Nk_Hstar_IL, 
                                                      H_star_label,
                                                      Nk_H )


        #
        # solve local problem 
        #
        
        if go_To_localProblem:
            counters['solve_cop'] += 1
            
            #
            # Solve Local problem for enlarging reseach region till it still retrive a counterfactual
            # retrive the last counterfactual solution computed
            #
            Hcop, Nk_Hcop, d_UHcop = Local_search_(counters, d,k, DataType,
                                                   data,features_type, features_multicat, U, S,
                                                   H_star, Nk_Hstar_IL,
                                                   epsilon, M, solver_opts, refinement)

            # Check if a better incumbent has been found
            if d_UHcop < d_UH:
                counters['better'] +=1 
                h = counters['better']
                t_newH_found = time.time() -t_start
                
                # new incumbent becomes the current one
                H, Nk_H, d_UH = copy.deepcopy(Hcop), copy.deepcopy(Nk_Hcop), copy.deepcopy(d_UHcop)
                
                # Evolution
                H_ev[h], Nk_H_ev[h], d_UH_ev[h], t_ev[h] = H, Nk_H, d_UH, t_newH_found
                
                # Relative Gap when available
                RelGap = rel_gap(d_UxS, d_UH) if use_rel_gap else None
                
            else:
                counters['worse']+=1
                g =  0                                              # GaussVNS restart
                print("Local problem retrived a further counterfactual than the incumbent.")
            
            
        # Current time
        t = time.time() - t_start
    
    ###########################################################################
    # Final refinement step
    ###########################################################################
    refinement = True if t > timelim else False
    if refinement:
        Href, Nk_Href, d_UHref = Local_search_(counters, d,k, DataType,
                                               data,features_type, features_multicat, U, S,
                                               H, Nk_H,
                                               epsilon, M, solver_opts, refinement)
         
        t_Href_found = time.time() -t_start          
        # Evolution
        H_ev[h+1], Nk_H_ev[h+1], d_UH_ev[h+1], t_ev[h+1] = Href, Nk_Href, d_UHref,t_Href_found

    ###########################################################################
    # Final refinement step
    ###########################################################################
    #
    #End while loop            
    #
    print_counters(counters)                
    print_evolution(H_ev, Nk_H_ev, d_UH_ev, t_ev, verbose=True)
    
    return H_ev, Nk_H_ev, d_UH_ev, t_ev, counters