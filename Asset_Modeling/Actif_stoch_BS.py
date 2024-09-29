'''2.1 Modele de Black et Scholes.
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

from Asset_Modeling.BM_def import BM
import numpy as np
import yfinance as yf
def simu_actif(init, t:int, T:int, mu, sigma, N=None):
    sigma_hourly = sigma/np.sqrt(365)
    mu_hourly = mu/365
    T = T*365
    t = t*365
    time_horizon = T - t

    if N is None:
        N = int(time_horizon * 24)

    if N <= 0:
        raise ValueError("Number of simulation steps (N) must be greater than 0.")

    St = [init]
    BM_ = BM(N, time_horizon)
    delta_t = time_horizon / N

    for i in range(N):
        BM_delta = BM_[i + 1] - BM_[i]
        St.append(St[i] * np.exp((mu_hourly - 0.5 * sigma_hourly ** 2) * delta_t + sigma_hourly * BM_delta))

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
    y = simu_actif(St, N, 0, N/365, mu, sigma)
    plt.plot(x, y)
    plt.legend(lengend)
    plt.show()

if __name__ == '__main__':
    for i in range(150):
        St = simu_actif(50, 0, 1/365, 0.2, 0.6)
        plt.plot(St)
    plt.show()