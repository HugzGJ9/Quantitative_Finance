import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

def handle_socket_message(msg):
    print(f"message type: {msg['e']}")
    print(msg)

def handle_dcm_message(depth_cache):
    print(f"symbol {depth_cache.symbol}")
    print("top 5 bids")
    print(depth_cache.get_bids()[:5])
    print("top 5 asks")
    print(depth_cache.get_asks()[:5])
    print("last update time {}".format(depth_cache.update_time))

def get_binance_orderbook(symbol)->list:
    market_depth = client.get_order_book(symbol=symbol)
    market_bids = pd.DataFrame(market_depth['bids'])
    market_bids.columns = ['price', 'bids']
    market_asks = pd.DataFrame(market_depth['asks'])
    market_asks.columns = ['price', 'asks']
    return [market_bids, market_asks]

def get_binance_last_trades(symbol):
    last_trades = client.get_recent_trades(symbol=symbol)
    last_trades = pd.DataFrame(last_trades)
    return last_trades

#def get_binance_options(symbol):
 #   options = client.option
if __name__ == '__main__':

    api_key = os.environ['BINANCE_API_KEY_TEST']
    api_secret = os.environ['BINANCE_API_SECRET_TEST']
    client = Client(api_key, api_secret, testnet=True)
    tickers = client.get_ticker()
    df_tickers = pd.DataFrame(tickers)
    btc_bids, btc_asks = get_binance_orderbook('BTCUSDT')
    btc_trades = get_binance_last_trades('BTCUSDT')
    #option = client.options_mark_price(symbol='BTC-221231-41000-C')


    # socket manager using threads
    twm = ThreadedWebsocketManager()
    twm.start()

    # depth cache manager using threads
    dcm = ThreadedDepthCacheManager()
    dcm.start()
    twm.start_kline_socket(callback=handle_socket_message, symbol='BTCUSDT')

    dcm.start_depth_cache(callback=handle_dcm_message, symbol='BTCUSDT')
    # replace with a current options symbol
    options_symbol = 'BTC-241227-38000-C'
    dcm.start_options_depth_cache(callback=handle_dcm_message, symbol=options_symbol)

    # join the threaded managers to the main thread
    twm.join()
    dcm.join()