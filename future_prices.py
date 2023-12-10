from statistics import mean

import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt
from Actif_stoch_BS import simu_actif, simu_stock_vs_hist
from interest_rates import Tresury_bond_5years


def future_price(St, t, T, r):
    Future_price = St*np.exp(r*(T-t))
    return Future_price
def investor_future_price(ticker_symbol, t, T, r):
    epsilone = 0 #corresponds to the systemic risk
    k = r + epsilone
    stock_data = yf.download(ticker_symbol, start='2022-01-01', end='2023-12-08')
    St = stock_data['Close'].iloc[0]
    stock_data['variation_daily'] = stock_data['Close'].pct_change()*100
    mu = (stock_data['variation_daily'].resample('Y').mean()).mean()
    sigma = (stock_data['variation_daily'].resample('Y').std()).mean() / 10
    simulations = []
    Nmc = 1000
    for i in range(Nmc):
        simulations.append(simu_actif(St, 100, 0, T, mu, sigma)[-1])
    ST_expected = mean(simulations)
    Future_price = ST_expected*np.exp((r-k)*(T-t))
    return Future_price

if __name__ == '__main__':

    print(investor_future_price('AAPL', 0, 2, Tresury_bond_5years))