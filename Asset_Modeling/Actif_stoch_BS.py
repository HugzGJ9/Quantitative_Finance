'''
Corrected and simplified implementation of the Black-Scholes model simulation,
realized volatility, and European option pricing (Monte Carlo).
'''

import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf

# Simulate an asset following the Black-Scholes model (Geometric Brownian Motion)
def simu_actif(S0, t, T, mu, sigma, N=None):
    """
    Simulate asset price St from time t to T under the Black-Scholes model.

    Parameters:
    S0 (float): Initial price
    t (float): Start time in years
    T (float): End time in years
    mu (float): Annual risk-free interest rate
    sigma (float): Annual volatility
    N (int): Number of time steps (optional, default daily)

    Returns:
    list: Simulated asset prices
    """
    if N is None:
        N = int((T - t) * 365)  # daily steps by default

    dt = (T - t) / N
    Wt = np.random.normal(0, np.sqrt(dt), size=N)
    Wt = np.insert(np.cumsum(Wt), 0, 0.0)

    time_grid = np.linspace(t, T, N + 1)
    St = S0 * np.exp((mu - 0.5 * sigma**2) * (time_grid - t) + sigma * Wt)

    return St.tolist()

# European call payoff
def payoff_call_eu(ST, K):
    return max(ST - K, 0)

# Monte Carlo pricing of a European call option
def option_eu_mc(S0, t, T, K, mu, sigma, Nmc=1000):
    payoff_sum = 0
    for _ in range(Nmc):
        prix_actif = simu_actif(S0, t, T, mu, sigma)
        payoff_sum += payoff_call_eu(prix_actif[-1], K)

    return payoff_sum / Nmc

# Simulate stock price vs historical market price
def simu_stock_vs_hist(ticker_symbol):
    legend = ['Market', 'Simulated']
    stock_data = yf.download(ticker_symbol, start='2022-01-01', end='2023-12-08')
    stock_data.Close.plot()

    daily_returns = stock_data['Close'].pct_change().dropna()
    mu = daily_returns.mean() * 365
    sigma = daily_returns.std() * np.sqrt(365)

    S0 = stock_data['Close'].iloc[0]
    N = len(stock_data['Close']) - 1
    T = N / 365

    simulated_prices = simu_actif(S0, 0, T, mu, sigma, N)
    plt.plot(stock_data.index, simulated_prices)

    plt.legend(legend)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Historical vs Simulated Stock Prices')
    plt.show()

# Calculate simple returns
def getreturns(St):
    return [St[i+1]/St[i] for i in range(len(St)-1)]

# Calculate log returns
def getlogreturns(St):
    returns = getreturns(St)
    return [np.log(r) for r in returns]

# Calculate realized volatility (sum of squared log returns)
def getvol(St, T):
    logreturns_squared = np.square(getlogreturns(St))
    return round(np.sqrt(1/T * np.sum(logreturns_squared)), 4)

# Main testing section
if __name__ == '__main__':
    S0 = 50
    mu = 0.2
    sigma = 0.6
    T = 1
    Nmc = 1000
    vols= []
    for i in range(Nmc):
        St = simu_actif(S0, 0, T, mu, sigma)
        vols.append(getvol(St, T))
        plt.plot(St)

    plt.title('Simulated Asset Price Paths (1 year)')
    plt.xlabel('Days')
    plt.ylabel('Asset Price')
    plt.show()
    print(f'AVG VOLS : {round(np.mean(vols), 4) * 100} %')
