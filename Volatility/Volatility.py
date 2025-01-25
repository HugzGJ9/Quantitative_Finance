import numpy as np

from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu
import datetime
import pandas as pd
import yfinance as yf
from toolbox import keep_after_dash, keep_before_dash
from Graphics.Graphics import plot_2d, correl_plot
import os
import matplotlib.pyplot as plt

def Volatilite_implicite(stock_name, maturity_date, option_type, r, plot=True, isCrypto=False):
    t = 0
    epsilon = 0.0001
    maturity = pd.Timestamp(maturity_date) - datetime.datetime.now()
    T = maturity.days / 365
    stock_obj = yf.Ticker(stock_name)
    S0 = stock_obj.history().tail(1)['Close'].values[0]

    if not isCrypto:
        options = stock_obj.option_chain(maturity_date)
        if option_type == "Call EU":
            options = options.calls
        elif option_type == "Put EU":
            options = options.puts
        options_df = options[['lastTradeDate', 'strike', 'lastPrice', 'impliedVolatility']]
    else:
        options_df = pd.read_excel('API/BTCUSD.xlsx')
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
        if options_df['lastPrice'].iloc[i] < S0 and options_df['lastPrice'].iloc[i] > max(S0 - int(options_df['strike'].iloc[i]) * np.exp(-r * T), 0):
            # Initialize sigma with a reasonable guess
            sigma = 0.6  # Or use a heuristic or known volatility value
            option_eu_obj = Option_eu(1, option_type, asset, options_df['lastPrice'].iloc[i], T, r, sigma)
            Market_price = options_df['lastPrice'].iloc[i]

            # Safeguard to avoid infinite loop or divergence
            max_iterations = 100
            iteration = 0

            # Newton-Raphson method for finding implied volatility
            while np.abs(option_eu_obj.option_price_close_formulae() - Market_price) > epsilon:
                vega = option_eu_obj.Vega_DF()

                if vega == 0:
                    print("Vega is zero, cannot continue.")
                    break

                # Update sigma using Newton's method
                sigma_new = sigma - (option_eu_obj.option_price_close_formulae() - Market_price) / vega

                # Ensure sigma stays within a reasonable range (e.g., no negative values)
                if sigma_new < 0:
                    sigma_new = 0.001  # or some small positive number

                sigma = sigma_new
                option_eu_obj = Option_eu(1, option_type, asset, options_df['lastPrice'].iloc[i], T, r, sigma)

                # Break loop if too many iterations
                iteration += 1
                if iteration >= max_iterations:
                    print("Maximum iterations reached.")
                    break

    plot_2d(strikes, vol_implicite, 'Strike', 'Implied volatility', plot=plot, title='Volatility smile')
    result = dict(zip(strikes, vol_implicite))
    return result

def volatilityReport(ticker='NG=F', ticker2=None):
    booking_file_path = f'Booking/Volatility_history_{ticker}.xlsx'
    booking_file_sheet_name = 'volatility'
    data_hourly = yf.download(ticker, start='2023-01-01', end='2024-09-22', interval='1h')

    if ticker2:
        data_hourly2 = yf.download(ticker2, start='2023-01-01', end='2024-09-22', interval='1h')
        data_hourly = data_hourly - data_hourly2
        booking_file_path = f'Booking/Volatility_history_spread_{ticker}-{ticker2}.xlsx'

    data_hourly = data_hourly.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
    data_hourly['return'] = data_hourly['Close'].pct_change()
    df_daily_vol = data_hourly.resample('D').std()*np.sqrt(252)*100
    df_weekly_vol = data_hourly.resample('W').std()*np.sqrt(252)*100
    data_hourly['4H_vol'] = data_hourly['return'].rolling(window=4).std()*np.sqrt(252)*100
    data_hourly = data_hourly.drop(columns=['Close'])
    df_daily_vol = df_daily_vol.drop(columns=['Close'])
    df_weekly_vol = df_weekly_vol.drop(columns=['Close'])
    data_hourly['date&time'] = data_hourly.index.tz_localize(None)
    df_daily_vol['date&time'] = df_daily_vol.index.tz_localize(None)
    df_weekly_vol['date&time'] = df_weekly_vol.index.tz_localize(None)

    with pd.ExcelWriter(booking_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        data_hourly.to_excel(writer, sheet_name=f'{booking_file_sheet_name} hourly', index=False)
        df_daily_vol.to_excel(writer, sheet_name=f'{booking_file_sheet_name} hourly to day', index=False)
        df_weekly_vol.to_excel(writer, sheet_name=f'{booking_file_sheet_name} hourly to week', index=False)

    start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    data_min = yf.download(ticker, start=start_date, end=end_date, interval='1m')
    if ticker2:
        data_min2 = yf.download(ticker2, start=start_date, end=end_date, interval='1m')
        data_min = data_min - data_min2
    data_min = data_min.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
    data_min['return'] = data_min['Close'].pct_change()
    data_min['30m_vol'] = data_min['return'].rolling(window=30).std() * np.sqrt(252*24*60) * 100
    data_min['1H_vol'] = data_min['return'].rolling(window=60).std() * np.sqrt(252*24*60) * 100
    data_min['2H_vol'] = data_min['return'].rolling(window=60*2).std() * np.sqrt(252*24*60) * 100
    data_min['4H_vol'] = data_min['return'].rolling(window=60*4).std() * np.sqrt(252*24*60) * 100
    data_min['8H_vol'] = data_min['return'].rolling(window=60*8).std() * np.sqrt(252*24*60) * 100
    data_min['16H_vol'] = data_min['return'].rolling(window=60*16).std() * np.sqrt(252*24*60) * 100
    data_min['24H_vol'] = data_min['return'].rolling(window=60*24).std() * np.sqrt(252*24*60) * 100
    data_min = data_min.drop(columns=['Close'])
    data_min['date&time'] = data_min.index.tz_localize(None)
    with pd.ExcelWriter(booking_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        data_min.to_excel(writer, sheet_name=f'{booking_file_sheet_name}', index=False)

    return data_hourly, data_min

def computecorr(ticker1, ticker2):
    data_hourly = yf.download(ticker1, start='2024-01-01', end=datetime.datetime.now().strftime("%Y-%m-%d"), interval='1h')
    data_hourly2 = yf.download(ticker2, start='2024-01-01', end=datetime.datetime.now().strftime("%Y-%m-%d"), interval='1h')
    # 1. Resample the data to monthly groups based on the index (datetime index)
    data_hourly['Close'].plot(label=ticker1)
    data_hourly2['Close'].plot(label=ticker2)
    plt.title(f"Price {ticker1} - {ticker2} evolution")
    plt.legend()

    plt.show()
    data_hourly['Month'] = data_hourly.index.to_period('M')
    data_hourly2['Month'] = data_hourly2.index.to_period('M')

    # 2. Merge the two DataFrames on index and the 'Month' column
    merged_df = pd.merge(data_hourly, data_hourly2, left_index=True, right_index=True, suffixes=('_left', '_right'))

    # 3. Group by the 'Month' and calculate the correlation for each month
    monthly_correlations = merged_df.groupby('Month_left')[['Close_left', 'Close_right']].corr().iloc[0::2, 1].reset_index()

    # Rename columns for clarity
    monthly_correlations.columns = ['Month', 'todrop', 'Correlation']
    monthly_correlations = monthly_correlations.drop(columns=['todrop'])
    correl_plot(monthly_correlations['Correlation'], 'month', 'Correlation', f'Monthly correlation evolution {ticker1} / {ticker2}')
    return monthly_correlations


if __name__ == '__main__':
    # vol = Volatilite_implicite('AAPL', '2024-12-20', 'Call EU', 0.04, plot=True, isCrypto=False)

    # NGF_hourly, NGF_min = volatilityReport('NG=F')
    # TTF_hourly, TTF_min = volatilityReport('TTF=F')

    # NGF_TTF_hourly, NGF_TTF_min = volatilityReport('NG=F', 'TTF=F')
    # print('hello')
    computecorr('NG=F', 'TTF=F')
    # merged_df = pd.merge(data_hourly, data_hourly2, left_index=True, right_index=True, suffixes=('_left', '_right'))