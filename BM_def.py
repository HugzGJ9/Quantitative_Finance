'''
Simulation du mouvement Brownien
    
On discr´etise l’intervalle [0, T] sur N intervalles Delta_t : (Delta_t = T/N, t_n = n*Delta_t, t_N = T).
Pour un chemin du mouvement Brownien on code:

W0 = 0
Wt1 = g1√Delta_t
Wt2 = Wt1 + g2√Delta_t
WtN = WtN−1 + gN√Delta_t
'''

import numpy as np
from Maths import norm_

def BM(N, T):
    delta_t = T/N
    W_t = [0]
    for i in range(N):
        W_t.append(W_t[i] + norm_() * np.sqrt(delta_t))
    return W_t