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
from matplotlib import pyplot as plt

from BM_def import BM
import numpy as np
import yfinance as yf

from Maths import norm_
def simu_actif(init, N, t, T, mu, sigma):
    St = [init]
    BM_ = BM(N, T-t)
    delta_t = (T-t)/N
    for i in range(N):
        BM_delta = BM_[i+1] - BM_[i]
        St.append(St[i]*np.exp((mu-0.5*sigma**2)*delta_t + sigma*BM_delta))
    return St

def payoff_call_eu(ST, K):
    return max(ST - K, 0)
def option_eu_mc(St, t, T, K, Nmc):
    prix_option = 0
    for i in range(Nmc):
        prix_actif = simu_actif(St, 100, t, T, 0.1, 0.3)
        prix_option += payoff_call_eu(prix_actif[-1], K)
    prix_option = prix_option/Nmc
    return prix_option
def simu_stock_vs_hist(ticker_symbol):
    lengend = ['Market', 'Simulated']
    stock_data = yf.download(ticker_symbol, start='2022-01-01', end='2023-12-08')
    stock_data.Close.plot()

    stock_data['variation_daily'] = stock_data['Close'].pct_change()*100
    mu = (stock_data['variation_daily'].resample('Y').mean()).mean()
    sigma = stock_data['variation_daily'].std()/10
    St = stock_data['Close'].iloc[0]
    N = stock_data['Close'].size - 1
    x = stock_data.index.to_list()
    y = simu_actif(St, N, 0, N/365.6, mu, sigma)
    plt.plot(x, y)
    plt.legend(lengend)
    plt.show()