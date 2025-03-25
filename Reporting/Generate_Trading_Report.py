from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from jinja2 import Template
import os
from Asset_Modeling.Actif_stoch_BS import getvol
from Template import HTML_TEMPLATE
from Charts import create_plotly_charts
def fetch_data(ticker, start_date, end_date, cache_dir="data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start_date.date()}_{end_date.date()}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=True)
    else:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = df.index
        if df.empty:
            print(f"No data retrieved for {ticker}. Check ticker or date range.")
            return pd.DataFrame()
        df.to_csv(cache_file)
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()


# Fetch intraday data
def fetch_intraday_data(ticker, interval="15m", period="30d"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = df.index
    if df.empty:
        print(f"No intraday data retrieved for {ticker} ({interval}).")
        return pd.DataFrame()
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Technical indicators
def add_technical_indicators(df):
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    return df


# Insights generation
def generate_insights(df):
    insights = []
    if df['RSI'].iloc[-1] > 70:
        insights.append("RSI indicates overbought conditions. Consider taking profits.")
    if df['RSI'].iloc[-1] < 30:
        insights.append("RSI indicates oversold conditions. Consider buying opportunities.")
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        insights.append("MACD indicates a bullish crossover.")
    if df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
        insights.append("MACD indicates a bearish crossover.")
    return insights


# KPIs calculation
def calculate_kpis(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    return {
        "Daily_Return": df['Daily_Return'].iloc[-1] * 100,
        "Cumulative_Return": df['Cumulative_Return'].iloc[-1] * 100,
        "Volatility": df['Daily_Return'].std() * np.sqrt(252) * 100,
        "Sharpe_Ratio": (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252),
        "Max_Drawdown": (df['Close'] / df['Close'].cummax() - 1).min() * 100
    }

def generate_html_report(df, kpis, insights, *figures, ticker, filename="report.html"):
    template = Template(HTML_TEMPLATE)
    filename = f"reports/trading_report_{ticker}.html"
    with open(filename, 'w', encoding='utf-8') as f:  # Add encoding here
        f.write(template.render(
            ticker=ticker,
            kpis=kpis[0],
            kpis_vol=kpis[1],
            insights=insights,
            figures=[fig.to_html(full_html=False, include_plotlyjs='cdn') for fig in figures],
            color= '#3498db'
        ))
    print(f"Report saved to {filename}")


def compute_volatility_metrics(df_daily, df_1h, df_15m, end_date):
    end_date = pd.Timestamp(end_date).tz_localize('UTC')
    vol_metrics = {}

    vol_metrics['vol_yearly'] = getvol(df_daily['Close'].to_list(), 252 / 365)

    vol_metrics['vol_monthly'] = getvol(df_1h['Close'].to_list(), 30 / 365)

    vol_metrics['vol_weekly'] = [
        getvol(
            df_1h[(df_1h['Date'] > (end_date - pd.Timedelta(days=x + 7))) &
                  (df_1h['Date'] < (end_date - pd.Timedelta(days=x)))]['Close'].to_list(),
            5 / 365
        )
        for x in range(0, 30, 7)
    ]

    vol_metrics['vol_daily'] = [
        getvol(
            df_1h[(df_1h['Date'] > (end_date - pd.Timedelta(days=x + 1))) &
                  (df_1h['Date'] < (end_date - pd.Timedelta(days=x)))]['Close'].to_list(),
            1 / 365
        )
        for x in range(0, 30)
    ]

    vol_metrics['vol_intraday_hourly'] = [
        getvol(
            df_15m[(df_15m['Date'] > (end_date - pd.Timedelta(hours=x + 1))) &
                   (df_15m['Date'] < (end_date - pd.Timedelta(hours=x)))]['Close'].to_list(),
            1 / (365 * 24)
        )
        for x in range(0, 24)
    ]

    return vol_metrics

def format_data(ticker):
    end_date = pd.Timestamp.now(tz='UTC')
    start_date = end_date - pd.Timedelta(days=365)
    # Data collection
    df = fetch_data(ticker, start_date, end_date)
    df_1h = fetch_intraday_data(ticker, interval="1h")
    df_15m = fetch_intraday_data(ticker, interval="15m")
    # Your exact volatility calculations
    vol_results = compute_volatility_metrics(df, df_1h, df_15m, end_date=datetime.today())
    # Prepare data
    df = add_technical_indicators(df)

    df_1h = add_technical_indicators(df_1h)
    df_15m = add_technical_indicators(df_15m)

    kpis = calculate_kpis(df)
    # Add volatility KPIs
    kpis_vol = {
        'Vol_Yearly': vol_results['vol_yearly'] * 100,
        'Vol_Monthly': vol_results['vol_monthly'] * 100,
        'Vol_Weekly_Latest': vol_results['vol_weekly'][0] * 100 if vol_results['vol_weekly'] else None,
        'Vol_Daily_Latest': vol_results['vol_daily'][0] * 100 if vol_results['vol_daily'] else None
    }
    # Create visualizations
    figures = create_plotly_charts(df, df_1h, df_15m, vol_results['vol_weekly'], vol_results['vol_daily'], vol_results['vol_intraday_hourly'])
    return df, figures, kpis, kpis_vol

def main(ticker):
    df, figures, kpis, kpis_vol = format_data(ticker)
    generate_html_report(df, [kpis, kpis_vol], generate_insights(df), *figures, ticker=ticker)

if __name__ == "__main__":
    main("GLE.PA")