import numpy as np

from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu
import datetime
import pandas as pd
import yfinance as yf
from toolbox import keep_after_dash, keep_before_dash
from Graphics.Graphics import plot_2d

def Volatilite_implicite(stock_name, maturity_date, option_type, r, plot=True, isCrypto=False):
    t = 0
    epsilon = 0.0001
    maturity = pd.Timestamp(maturity_date) - datetime.now()
    T = maturity.days / 365.6
    stock_obj = yf.Ticker(stock_name)
    S0 = stock_obj.history().tail(1)['Close'].values[0]

    if not isCrypto:
        options = stock_obj.option_chain(maturity_date)
        if option_type == "Call EU":
            options = options.calls
        elif option_type == "Put EU":
            options = options.puts
        options_df = options[['lastTradeDate', 'strike', 'bid', 'impliedVolatility']]
    else:
        options_df = pd.read_excel('Crypto/BTCUSD.xlsx')
        options_df['matu'] = options_df['strike'].apply(keep_before_dash)
        options_df = options_df[options_df['matu'] == pd.Timestamp(maturity_date)]
        options_df['strike'] = options_df['strike'].apply(keep_after_dash)
        options_df = options_df[options_df['volume_contracts'] > 0.0]
        if option_type == "Call EU":
            options_df = options_df[options_df['type']=='C']
        elif option_type == "Put EU":
            options_df = options_df[options_df['type']=='P']

        options_df['bid'] = options_df['best_bid_price']
        options_df['ask'] = options_df['best_ask_price']

        options_df = options_df[['strike', 'bid', 'ask']]
        options_df = options_df.groupby('strike').mean()
        options_df['strike'] = options_df.index

    asset = asset_BS(S0, 0)
    vol_implicite = []
    strikes = []
    for i in range(len(options_df)):
        if options_df['bid'].iloc[i] < S0 and options_df['bid'].iloc[i] > max(S0 - int(options_df['strike'].iloc[i]) * np.exp(-r * T), 0):
            sigma = np.sqrt(2 * np.abs(np.log(r * T + S0 / int(options_df['strike'].iloc[i]))) / T)
            option_eu_obj = Option_eu(1, option_type, asset, int(options_df['strike'].iloc[i]), t, T, r, sigma)
            Market_price = options_df['bid'].iloc[i]
            # Algorithme de Newton :
            while np.abs(option_eu_obj.option_price_close_formulae() - Market_price) > epsilon:
                sigma = sigma - (option_eu_obj.option_price_close_formulae() - Market_price) / option_eu_obj.Vega_DF()
                option_eu_obj = Option_eu(1, option_type, asset, int(options_df['strike'].iloc[i]), t, T, r, sigma)

            strikes.append(int(options_df['strike'].iloc[i]))
            vol_implicite.append(sigma)

    plot_2d(strikes, vol_implicite, 'Volatility smile', 'Strike', 'Implied volatility', isShow=plot)
    result = dict(zip(strikes, vol_implicite))
    return result