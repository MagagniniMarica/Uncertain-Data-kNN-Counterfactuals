# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:47:32 2026

@author: Marica Magagnini
"""

from pyomo import environ as pym
import pandas as pd


# d indicates dhe distance to construct Nk (contraints level)
# d_obj indicates the shape of the obj function 
def COP_(DataType,d, d_obj, data, features, features_type, U, S,N, J, k, M, eps):
    
    #
    # Model definition
    #
    m = pym.ConcreteModel(name = 'Counterfactual Optimization Problem')
    
    #
    # Indexes and Parameters
    #
    m.j = pym.Set(initialize = features)                    # All features
 
    
    # Number of instances  
    m.n = pym.RangeSet(0,N-1)                               # 0,...,N-1
    
    
    # Target variable
    labels = pd.DataFrame([int(data[n].label) for n in range(N)], columns=['label'] )
    m.yn = pym.Param(m.n, initialize=labels)

    
    # label dependent indices
    
    if d=='d0': # m.n0= m.n1 = m.n #no label-sensitive distances
        m.n0=  pym.RangeSet(0,N-1) 
    else:
        n0_ = [i  for i in range(N) if labels['label'][i] == 0]
        m.n0=  pym.Set(initialize = n0_)
        n1_ = [i  for i in range(N) if labels['label'][i] == 1]
        m.n1= pym.Set(initialize = n1_)
    
    
    if DataType =='R':
        #Extreams: Un = prod_{j}{[a^n_j,b^n_j]}
        Una = pd.DataFrame([{f: data[n].bounds()[f][0] for f in features} for n in range(N)])
        Unb = pd.DataFrame([{f: data[n].bounds()[f][1] for f in features} for n in range(N)])
        def an_init(m,n,j):
            return Una[j][n]
        m.Una = pym.Param(m.n,m.j, initialize=an_init,mutable=True)
        def bn_init(m,n,j):
            return Unb[j][n]
        m.Unb = pym.Param(m.n,m.j, initialize=bn_init, mutable=True)
        
        # U
        Ua_={f: U.bounds()[f][0] for f in features}
        Ub_ = {f: U.bounds()[f][1] for f in features}
        m.Ua = pym.Param(m.j, initialize = Ua_ )
        m.Ub = pym.Param(m.j, initialize = Ub_)
        
        # S
        Sa_ ={f: S.bounds()[f][0] for f in features}
        Sb_ = {f: S.bounds()[f][1] for f in features}
        m.Sa = pym.Param(m.j, initialize = Sa_ )
        m.Sb = pym.Param(m.j, initialize = Sb_)
        
    
    elif DataType=='B':
        # Ball centers Un = {u : ||u-xn|| <= rn }
        dataset = pd.DataFrame([{f: data[n].x[f] for f in features} for n in range(N)])
        def xn_init(m,n,j):
            return dataset[j][n]
        m.cUn = pym.Param(m.n,m.j, initialize=xn_init, mutable=True)
        # Radius
        r = pd.DataFrame([data[n].r for n in range(N)], columns= ['radius'] )
        m.Unr = pym.Param(m.n, initialize=r)
        
        # U
        Uc = U.x
        m.Uc = pym.Param(m.j, initialize = Uc )
        m.Ur = pym.Param(initialize = U.r)
        
        # S
        Sc = S.x
        m.cS =  pym.Param(m.j, initialize = Sc)
        m.Sr = pym.Param( initialize = S.r)

    #
    # other parameters depending on dint2
    #   
    if d == 'dint2':
        
        #nu^Un
        def nun_init(m,n):
            if DataType == 'R':
                return (1/12)* sum( (Unb[j][n]- Una[j][n])**2 + (Sb_[j] - Sa_[j])**2 for j in m.j)
            elif DataType == 'B':
                return (J/(J+2)) * (data[n].r**2 + S.r**2)
        m.nun = pym.Param(m.n1, initialize = nun_init)
        
        
        
        # mu^Un
        def muUn_init(m,n,j):
            if DataType == 'R':
                return (Unb[j][n]+ Una[j][n])/2
            elif DataType == 'B':
                return dataset[j][n]
        m.mun = pym.Param(m.n1, m.j, initialize = muUn_init, mutable = True)
    
    if d == 'dint2' or d_obj == 'dint2' :
        # mu^S
        def muS_init(m,j):
            if DataType == 'R':
                return (Sb_[j] + Sa_[j])/2
            elif DataType == 'B':
                return Sc[j]
        m.muS = pym.Param(m.j, initialize = muS_init)
    if d_obj == 'dint2':
        # mu^U
        def muU_init(m,j):
            if DataType == 'R':
                return (Ub_[j] + Ua_[j])/2
            elif DataType == 'B':
                return Uc[j]
        m.muU = pym.Param(m.j, initialize = muU_init)
        
        # nu^U
        if DataType == 'R': 
            m.nuU = pym.Param(initialize = (1/12)* sum( (Ub_[j]-  Ua_[j])**2 + (Sb_[j] - Sa_[j])**2 for j in m.j))
        elif DataType == 'B':
            m.nuU = pym.Param(initialize = (J/(J+2)) * (U.r**2 + S.r**2))
    
    #
    # variables
    #
    def x_domain_rule(m, j):
        if features_type[j] == 'Numerical':
            return pym.Reals
        else:  # 'category'
            return pym.Binary
    m.x = pym.Var(m.j, within=x_domain_rule,  bounds= (-30,30))# che bounds? Counterfactual center
    m.sk = pym.Var( within=pym.Reals,  bounds= (0,M)) # sk = rk^2
    m.v = pym.Var(m.n, within=pym.Binary)
    m.w = pym.Var(m.n, within=pym.Binary)
    
    
    #
    # Other variables depending on d/d_objand DataType
    #
    # constr vars
    
    if DataType == 'R' :
        if d != 'dint2':
            m.t = pym.Var(m.n,m.j, within=pym.NonNegativeReals, bounds= (0,M)) 
            m.z1 = pym.Var(m.n, m.j, within = pym.Binary)
            m.z2 = pym.Var(m.n,m.j, within = pym.Binary) 
        else:
            m.t = pym.Var(m.n0,m.j, within=pym.NonNegativeReals, bounds= (0,M)) 
            m.z1 = pym.Var(m.n0, m.j, within = pym.Binary)
            m.z2 = pym.Var(m.n0,m.j, within = pym.Binary) 
        
        # defined only for the instances that use d0
        m.z3 = pym.Var(m.n0,m.j, within = pym.Binary) 
    elif  DataType == 'B':
        if d != 'dint2':
            m.t_bar = pym.Var(m.n, within=pym.NonNegativeReals, bounds= (0,M)) 
        else:
            m.t_bar = pym.Var(m.n0, within=pym.NonNegativeReals, bounds= (0,M)) 
        # defined only for the instances that use d0 
        m.t = pym.Var(m.n0, within=pym.NonNegativeReals, bounds= (0,M))  
        m.z1 = pym.Var(m.n0, within = pym.Binary)
        m.z2 = pym.Var(m.n0, within = pym.Binary) 
            
    # obj vars
    if d_obj != 'dint2':
        if DataType == 'R' :
            m.T = pym.Var(m.j, within=pym.NonNegativeReals, bounds= (0,M)) 
            m.Z1 = pym.Var(m.j, within = pym.Binary)
            m.Z2 = pym.Var(m.j, within = pym.Binary) 
            if d_obj== 'd0': 
                m.Z3 = pym.Var(m.j, within = pym.Binary) 
        elif  DataType == 'B':
            m.T_bar = pym.Var( within=pym.NonNegativeReals, bounds= (0,M))
            if d_obj== 'd0':
                m.T = pym.Var( within=pym.NonNegativeReals, bounds= (0,M)) 
                m.Z1 = pym.Var(within = pym.Binary)
                m.Z2 = pym.Var(within = pym.Binary) 
    
    ###########################################################################
    # Objective function 
    ###########################################################################
    
    
    
    def objfunction_(m):
        if d_obj == 'dint2':
            return    pym.quicksum( (m.x[j] - m.muU[j] + m.muS[j])**2 for j in m.j) + m.nuU
        elif d_obj == 'd0':
            if DataType == 'R':
                return pym.quicksum( m.T[j]**2 for j in m.j)
            if DataType == 'B':
                return m.T**2
        elif d_obj == 'd1':
            if DataType == 'R':
                return pym.quicksum( m.T[j] for j in m.j)
            if DataType == 'B':
                return (m.T_bar +   m.Sr + m.Ur )**2 
        

    m.objfunction = pym.Objective(rule=objfunction_,  sense=pym.minimize)   
    
   
        
    ###########################################################################
    # Constraint definition and declaration
    ###########################################################################
    
    
    #
    # Constraints patterns
    #

    if DataType == 'R':
        def abx_(m,n,j):
            return m.Una[n,j] - m.Sb[j] - m.x[j]
        def xba_(m,n,j):
            return m.x[j] - m.Unb[n,j] + m.Sa[j]
    elif DataType == 'B':
        def NORM_x_cUncS_2(m,n):
            return pym.quicksum( (m.x[j] - m.cUn[n,j] + m.cS[j])**2 for j in m.j)
    if d == 'dint2':
        def x_mU_mS_nu_(m,n):
            return pym.quicksum( (m.x[j] - m.mun[n,j] + m.muS[j])**2 for j in m.j) + m.nun[n]
            
    
     
    #
    # counterfactual rule
    #
    def countRule_(m,):
        return pym.quicksum(m.w[n]*m.yn[n] for n in m.n) >= (1/2)*(pym.quicksum(m.w[n] for n in m.n) +1)
    m.countRule = pym.Constraint( rule= countRule_)
    
    #
    #neighborhood cardinality
    #
    def Nkw_(m,):
        return pym.quicksum(m.w[n] for n in m.n) >= k
    m.Nkw = pym.Constraint(rule=Nkw_)
    def Nkv_(m,):
        return pym.quicksum(m.v[n] for n in m.n) <= k-1 
    m.Nkv = pym.Constraint(rule=Nkv_)
    
    #
    #neighborhood composition
    #
    if DataType == 'R':
        #Constraints 15(c)
        def _15c_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return pym.quicksum(m.t[n,j] for j in m.j) - M*(1-m.w[n]) <= m.sk
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) - M*(1-m.w[n]) <= m.sk
            else: #d0
                return pym.quicksum(m.t[n,j]**2 for j in m.j) - M*(1-m.w[n]) <= m.sk
        m._15c = pym.Constraint(m.n, rule= _15c_, name='(15c)')
        
        #Constraints 15(d)
        def _15d_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return pym.quicksum(m.t[n,j] for j in m.j) + M*m.w[n] >= m.sk + eps
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) + M*m.w[n] >= m.sk + eps
            else:#d0
                return pym.quicksum(m.t[n,j]**2 for j in m.j)+ M*m.w[n] >= m.sk + eps
        m._15d = pym.Constraint(m.n, rule= _15d_, name='(15d)')    
        
        #Constraints 15(e)
        def _15e_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return pym.quicksum(m.t[n,j] for j in m.j) - M*(1-m.v[n]) <= m.sk - eps
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) - M*(1-m.v[n]) <= m.sk - eps
            else: #d0
                return pym.quicksum(m.t[n,j]**2 for j in m.j) - M*(1-m.v[n]) <= m.sk - eps
        m._15e = pym.Constraint(m.n, rule= _15e_, name='(15e)')      
        
        #Constraints 15(f)
        def _15f_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return pym.quicksum(m.t[n,j] for j in m.j) + M*m.v[n] >= m.sk 
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) + M*m.v[n] >= m.sk 
            else:#d0
                return pym.quicksum(m.t[n,j]**2 for j in m.j) + M*m.v[n] >= m.sk 
        m._15f = pym.Constraint(m.n, rule= _15f_, name='(15f)')   
    
    elif DataType == 'B':
        
        #Constraints 15(c)
        def _15c_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return (m.t_bar[n] +  m.Sr + m.Unr[n])**2 - M*(1-m.w[n]) <= m.sk
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) - M*(1-m.w[n]) <= m.sk
            else:#d0
                return m.t[n]**2 - M*(1-m.w[n]) <= m.sk
        m._15c = pym.Constraint(m.n, rule= _15c_, name='(15c)')
        
        #Constraints 15(d)
        def _15d_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return (m.t_bar[n] +  m.Sr + m.Unr[n])**2 + M*m.w[n] >= m.sk + eps
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) + M*m.w[n] >= m.sk + eps
            else:#d0
                return m.t[n]**2 + M*m.w[n] >= m.sk + eps
        m._15d = pym.Constraint(m.n, rule= _15d_, name='(15d)')    
        
        #Constraints 15(e)
        def _15e_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return (m.t_bar[n] +  m.Sr + m.Unr[n])**2 - M*(1-m.v[n]) <= m.sk - eps
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) - M*(1-m.v[n]) <= m.sk - eps
            else:#d0
                return m.t[n]**2 - M*(1-m.v[n]) <= m.sk - eps
        m._15e = pym.Constraint(m.n, rule= _15e_, name='(15e)')      
        
        #Constraints 15(f)
        def _15f_(m,n):
            if labels['label'][n] == 1 and d == 'd1':
                return (m.t_bar[n] +  m.Sr + m.Unr[n])**2 + M*m.v[n] >= m.sk 
            elif labels['label'][n] == 1 and d == 'dint2':
                return x_mU_mS_nu_(m,n) + M*m.v[n] >= m.sk 
            else:#d0
                return m.t[n]**2 + M*m.v[n] >= m.sk 
        m._15f = pym.Constraint(m.n, rule= _15f_, name='(15f)')
    
     
    #
    # Additional contraints for label-sensitive distance definition  (no additional constraints for dint2)
    #
    
    # d0 - always present, at least associated to 0-label instances
    # declaration using index m.n0: m.n0 = m.n if d = 'd0', m.n0 = {n : yn = 0} otherwise
    if DataType == 'R':
        def c_tz1_l_(m,n,j):
            return abx_(m,n,j) <= m.t[n,j]
        m.c_tz1_l_d0 = pym.Constraint(m.n0,m.j, rule= c_tz1_l_, name='c_tz1_l_R_d0')
        def c_tz1_u_(m,n,j):
            return m.t[n,j] <=  abx_(m,n,j) + M*(1-m.z1[n,j])
        m.c_tz1_u_d0 = pym.Constraint(m.n0,m.j, rule= c_tz1_u_, name='c_tz1_u_R_d0')
        def c_tz2_(m,n,j): # tn are already non negative
            return m.t[n,j] <=  M*(1-m.z2[n,j])
        m.c_tz2_d0 = pym.Constraint(m.n0,m.j, rule= c_tz2_, name='c_tz2_R_d0')
        def c_tz3_l_(m,n,j):
            return xba_(m,n,j) <= m.t[n,j]
        m.c_tz3_l_d0 = pym.Constraint(m.n0,m.j, rule= c_tz3_l_, name='c_tz3_l_R_d0')
        def c_tz3_u_(m,n,j):
            return m.t[n,j] <= xba_(m, n, j) + M*(1-m.z3[n,j])
        m.c_tz3_u_d0 = pym.Constraint(m.n0,m.j, rule= c_tz3_u_, name='c_tz3_u_R_d0')
        def c_z_(m,n,j):
            return m.z1[n,j] + m.z2[n,j] + m.z3[n,j] == 1
        m.c_z_d0 = pym.Constraint(m.n0,m.j, rule= c_z_, name='c_z_R_d0')
    
    elif DataType =='B':
        def c_tz1_l_(m,n):
            return m.t_bar[n] - m.Unr[n] - m.Sr <= m.t[n]
        m.c_tz1_l_d0 = pym.Constraint(m.n0, rule= c_tz1_l_, name='c_tz1_l_B_d0')
        def c_tz1_u_(m,n):
            return m.t[n] <= m.t_bar[n] - m.Unr[n] - m.Sr + M*(1- m.z1[n])
        m.c_tz1_u_d0 = pym.Constraint(m.n0, rule= c_tz1_u_, name='c_tz1_u_B_d0')
        def c_tz2_(m,n): # tn are already non negative
            return m.t[n] <= M*(1- m.z2[n])
        m.c_tz2_d0 = pym.Constraint(m.n0, rule= c_tz2_, name='c_tz2_B_d0')
        def c_tbar2_eq_norm2_(m,n):
            return m.t_bar[n]**2 == NORM_x_cUncS_2(m,n)
        m.c_tbar2_eq_norm2_d0 = pym.Constraint(m.n0, rule= c_tbar2_eq_norm2_, name='c_tbar2_eq_norm2_B_d0')
        def c_z_(m,n):
            return m.z1[n] + m.z2[n] == 1
        m.c_z_d0 = pym.Constraint(m.n0, rule= c_z_, name='c_z_B_d0')
    
    
    

    # d1 - to declare only for 1-labeled instances in case of d = 'd1'
    # use m.n1 index to declare   : m.n1 = {n : yn = 1}
    if d == 'd1':
        if DataType == 'R':
            def c_d1_tz1_l_(m,n,j):
                return abx_(m,n,j)**2 <= m.t[n,j]
            m.c_tz1_l_d1 = pym.Constraint(m.n1,m.j, rule= c_d1_tz1_l_, name='c_tz1_l_R_d1')
            def c_d1_tz1_u_(m,n,j):
                return m.t[n,j] <=  abx_(m,n,j)**2 + M*(1-m.z1[n,j])
            m.c_tz1_u_d1 = pym.Constraint(m.n1,m.j, rule= c_d1_tz1_u_, name='c_tz1_u_R_d1')
            def c_d1_tz2_l_(m,n,j):
                return xba_(m,n,j)**2 <= m.t[n,j]
            m.c_tz2_l_d1 = pym.Constraint(m.n1,m.j, rule= c_d1_tz2_l_, name='c_tz2_l_R_d1')
            def c_d1_tz2_u_(m,n,j):
                return  m.t[n,j] <= xba_(m, n, j)**2 + M*(1-m.z2[n,j])
            m.c_tz2_u_d1 = pym.Constraint(m.n1,m.j, rule= c_d1_tz2_u_, name='c_tz2_u_R_d1')
            def c_d1_z_(m,n,j):
                return m.z1[n,j] + m.z2[n,j] == 1
            m.c_z_d1 = pym.Constraint(m.n1,m.j, rule= c_d1_z_, name='c_z_R_d1')
        
        elif DataType =='B':

            def c_d1_tbar2_eq_norm2_(m,n):
                return m.t_bar[n]**2 == NORM_x_cUncS_2(m,n)
            m.c_tbar2_eq_norm2_d1 = pym.Constraint(m.n1, rule= c_d1_tbar2_eq_norm2_, name='c_tbar2_eq_norm2_B_d1')

    
    
    #
    # Similarly for obj function : Additional contraints for label-sensitive distance definition  (no additional constraints for dint2)
    #
    
    if d_obj == 'd0':
        if DataType == 'R':
            def c_objRd0_TZ1_l_(m,j):
                return   m.Ua[j] -m.Sb[j] - m.x[j] <= m.T[j]
            m.c_objRd0_TZ1_l = pym.Constraint(m.j, rule= c_objRd0_TZ1_l_, name='c_objRd0_TZ1_l')
            def c_objRd0_TZ1_u_(m,j):
                return m.T[j] <=  m.Ua[j] -m.Sb[j] - m.x[j]   + M*(1-m.Z1[j])
            m.c_objRd0_TZ1_u = pym.Constraint(m.j, rule= c_objRd0_TZ1_u_, name='c_objRd0_TZ1_u')
            def c_objRd0_TZ2_(m,j): #  T already non negative
                return m.T[j] <=  M*(1-m.Z2[j])
            m.c_objRd0_TZ2 = pym.Constraint(m.j, rule= c_objRd0_TZ2_, name='c_objRd0_TZ2')
            def c_objRd0_TZ3_l_(m,j):
                return m.x[j]- m.Ub[j] + m.Sa[j] <= m.T[j]
            m.c_objRd0_TZ3_l = pym.Constraint(m.j, rule= c_objRd0_TZ3_l_, name='c_objRd0_TZ3_l')
            def c_objRd0_TZ3_u_(m,j):
                return m.T[j] <= m.x[j]- m.Ub[j] + m.Sa[j] + M*(1-m.Z3[j])
            m.c_objRd0_TZ3_u = pym.Constraint(m.j, rule= c_objRd0_TZ3_u_, name='c_objRd0_TZ3_u')
            def c_objRd0_z_(m,j):
                return m.Z1[j] + m.Z2[j] + m.Z3[j] == 1
            m.c_objRd0_z = pym.Constraint(m.j, rule= c_objRd0_z_, name='c_objRd0_z')
            
        if DataType == 'B':
            m.c_obj_d0_B_TZ1l = pym.Constraint(expr= m.T_bar - m.Ur - m.Sr <= m.T)
            m.c_obj_d0_B_TZ1u = pym.Constraint(expr= m.T <= m.T_bar - m.Ur - m.Sr + M*(1- m.Z1))                                 
            m.c_obj_d0_B_TZ2 = pym.Constraint(expr= m.T <= M*(1- m.Z2))
            m.c_obj_d0_B_TbarNorm = pym.Constraint(expr= m.T_bar**2 == pym.quicksum( (m.x[j] - m.Uc[j] + m.cS[j])**2 for j in m.j))
            m.c_obj_d0_B_z = pym.Constraint(expr=m.Z1 + m.Z2 == 1)
            
    elif d_obj == 'd1':
        if DataType == 'R':
            def c_objRd1_TZ1_l_(m,j):
                return (m.Ua[j] -m.Sb[j] - m.x[j])**2 <= m.T[j]
            m.c_objRd1_TZ1_l = pym.Constraint(m.j, rule= c_objRd1_TZ1_l_, name='c_objRd1_TZ1_l')
            def c_objRd1_TZ1_u_(m,j):
                return m.T[j] <=  (m.Ua[j] -m.Sb[j] - m.x[j])**2 + M*(1-m.Z1[j])
            m.c_objRd1_TZ1_u = pym.Constraint(m.j, rule= c_objRd1_TZ1_u_, name='c_objRd1_TZ1_u')
            def c_objRd1_TZ2_l_(m,j):
                return (m.x[j]- m.Ub[j] + m.Sa[j])**2 <= m.T[j]
            m.c_objRd1_TZ2_l = pym.Constraint(m.j, rule= c_objRd1_TZ2_l_, name='c_objRd1_TZ2_l')
            def c_objRd1_TZ2_u_(m,j):
                return  m.T[j] <= (m.x[j]- m.Ub[j] + m.Sa[j])**2 + M*(1-m.Z2[j])
            m.c_objRd1_TZ2_u = pym.Constraint(m.j, rule= c_objRd1_TZ2_u_, name='c_objRd1_TZ2_u')
            def c_objRd1_Z_(m,j):
                return m.Z1[j] + m.Z2[j] == 1
            m.c_objRd1_Z = pym.Constraint(m.j, rule= c_objRd1_Z_, name='c_objRd1_Z')
            
        if DataType == 'B':
            m.c_obj_d1_B = pym.Constraint(expr= m.T_bar**2 == pym.quicksum( (m.x[j] - m.Uc[j] + m.cS[j])**2 for j in m.j))

            
    
    ###########################################################################
    ###########################################################################


   
    
    return m