
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from binance.client import Client


def get_binance_orderbook(symbol):
    market_depth = client.get_order_book(symbol=symbol)
    market_bids = pd.DataFrame(market_depth['bids'])
    market_bids.columns = ['price', 'bids']
    market_asks = pd.DataFrame(market_depth['asks'])
    market_asks.columns = ['price', 'asks']
    return [market_bids, market_asks]

if __name__ == '__main__':

    api_key = os.environ['BINANCE_API_KEY_TEST']
    api_secret = os.environ['BINANCE_API_SECRET_TEST']
    client = Client(api_key, api_secret, testnet=True)
    tickers = client.get_ticker()
    df_tickers = pd.DataFrame(tickers)
    btc_bids, btc_asks = get_binance_orderbook('BTCUSDT')

