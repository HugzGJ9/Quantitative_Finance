'''2.1 Mod`ele de Black et Scholes.
L’actif sous-jacent St est un processus stochastique S(t), t ∈ [t, T] qui verifie l’´equation différentielle
stochastique suivante:

dSt = St(\mu dt + \sigma dWt)
St = S0
mu est le taux d'interet sans risque
sigma est la volatilité de l'action

la solution de l'EDS est donnée par :
S(t) = S(0)*exp((mu-0.5sigma**2)$t + sigmaWt)
'''

from BM_def import BM
import numpy as np
from Maths import norm
def simu_actif(init, N, T, mu, sigma):
    St = [init]
    BM_ = BM(N, T)
    delta_t = T/N
    for i in range(N):
        BM_delta = BM_[i+1] - BM_[i]
        St.append(St[i]*np.exp((mu-0.5*sigma**2)*delta_t + sigma*BM_delta))
    return St