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
from Asset_Modeling.Maths import norm_
import matplotlib.pyplot as plt

def BrownianMotion(N, T):
    delta_t = T/N
    W_t = [0]
    for i in range(N):
        W_t.append(W_t[i] + norm_() * np.sqrt(delta_t))
    return W_t

def BrownianBridge(N, T):
    W_t = BrownianMotion(N, T)
    t = np.linspace(0, T, N + 1)
    X_t = W_t - (t / T) * W_t[-1]
    return X_t

def MC_simulations(Nmc:int =1000, funct=BrownianMotion):
    N, T = 24, 1/365
    for i in range(Nmc):
        path = funct(N, T)
        plt.plot(np.linspace(0, T, N + 1), path)
    plt.title('Simulations')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
if __name__ == '__main__':
    MC_simulations(Nmc=10000, funct=BrownianMotion)
    MC_simulations(Nmc=10000, funct=BrownianBridge)