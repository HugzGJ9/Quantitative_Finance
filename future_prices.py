import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt

from Actif_stoch_BS import simu_actif
def get_future_prices(ticker_symbol, maturities):

    stock_data = yf.download(ticker_symbol, start='2022-01-01', end='2023-12-08')
    plt.plot((stock_data.Close)


def future_price(St, t, T, r):
    epsilone = 0 #corresponds to the systemic risk
    k = r + epsilone
    ST_expected = simu_actif(St, 100, t, T, 0.1, 0.3)
    Future_price = ST_expected*np.exp((r-k)*(T-t))
    return

if __name__ == '__main__':
    get_future_prices('AAPL', 'maturities')