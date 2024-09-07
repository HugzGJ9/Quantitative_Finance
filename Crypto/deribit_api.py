""" Download CC-Option Data from Deribit via public API """

"""
code from :
https://github.com/bottama/Deribit-Option-Data/blob/main/deliverable/option-data-download.py
Matteo Bottacini -- matteo.bottacini@usi.ch
Last Update: February 21, 2021


"""

# import modules
import json
import requests
import pandas as pd
from tqdm import tqdm
import sqlite3
import datetime
from datetime import datetime
from datetime import date

# functions
def get_option_name_and_settlement(coin):
    """
    :param coin: crypto-currency coin name ('BTC', 'ETH')
    :return: 2 lists:
                        1.  list of traded options for the selected coin;
                        2.  list of settlement period for the selected coin.
    """

    # requests public API
    r = requests.get("https://test.deribit.com/api/v2/public/get_instruments?currency=" + coin + "&kind=option")
    result = json.loads(r.text)

    # get option name
    name = pd.json_normalize(result['result'])['instrument_name']
    name = list(name)

    # get option settlement period
    settlement_period = pd.json_normalize(result['result'])['settlement_period']
    settlement_period = list(settlement_period)

    return name, settlement_period


def get_option_data(coin):
    """
    :param coin: crypto-currency coin name ('BTC', 'ETH')
    :return: pandas data frame with all option data for a given coin
    """

    # get option name and settlement
    coin_name = get_option_name_and_settlement(coin)[0]
    settlement_period = get_option_name_and_settlement(coin)[1]

    # initialize data frame
    coin_df = []

    # initialize progress bar
    pbar = tqdm(total=len(coin_name))

    # loop to download data for each Option Name
    for i in range(len(coin_name)):
        # download option data -- requests and convert json to pandas
        r = requests.get('https://test.deribit.com/api/v2/public/get_order_book?instrument_name=' + coin_name[i])
        result = json.loads(r.text)
        df = pd.json_normalize(result['result'])

        # add settlement period
        df['settlement_period'] = settlement_period[i]

        # append data to data frame
        coin_df.append(df)

        # update progress bar
        pbar.update(1)

    # finalize data frame
    coin_df = pd.concat(coin_df)

    # remove useless columns from coin_df
    columns = ['state', 'estimated_delivery_price']
    coin_df.drop(columns, inplace=True, axis=1)

    # close the progress bar
    pbar.close()

    return coin_df

def get_option_prices(Currency, type, maturity):
    df = get_option_data(Currency)
    split_instrument_name = df['instrument_name'].str.split('-')
    df_option_prices = pd.DataFrame()
    df_option_prices['instrument_name'] = df['instrument_name']
    df_option_prices['type'] = split_instrument_name.str[3]
    df_option_prices['strike'] = split_instrument_name.str[2]
    df_option_prices['maturity'] = pd.to_datetime(split_instrument_name.str[1])
    df_option_prices['mark_iv'] = df['mark_iv']
    df_option_prices['bid'] = df['best_bid_price'] * df['underlying_price']
    df_option_prices['ask'] = df['best_ask_price'] * df['underlying_price']

    df_option_prices = df_option_prices[df_option_prices['type'] == type]
    df_option_prices = df_option_prices[df_option_prices['maturity'] == maturity]
    df_option_prices.index = df_option_prices['instrument_name']
    df_option_prices = df_option_prices.drop('instrument_name', axis=1)

    return df_option_prices
if __name__ == '__main__':

    # print data and time for log
    print('Date and time: ' +  datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' , format: dd/mm/yyyy hh:mm:ss')

    # download data -- BTC and ETH Options
    # btc_data = get_option_data('BTC')
    # eth_data = get_option_data('ETH')
    option_prices = get_option_prices('BTC', 'C', '2024-12-27')
    print('end')
    # export data to .csv -- append to existing
    # btc_data.to_csv('../csv_files/btc_option_data.csv', index=0, mode='a', header=False)
    # eth_data.to_csv('../csv_files/eth_option_data.csv', index=0, mode='a', header=False)

    # transform each element of the data frames into strings for sqlite3
    # btc_data = btc_data.astype(str)
    # eth_data = eth_data.astype(str)

    # connect to the SQLite3 database -- option_data.db
    # conn = sqlite3.connect('option-data.db')
    # print('Connection established with SQLite3 server: option-data.db')

    # create/update BTC and ETH tables in the database
    # btc_data.to_sql(name='btc_option_data', con=conn, if_exists='append', chunksize=None, index=False)
    # print('BTC data appended')

    # eth_data.to_sql(name='eth_option_data', con=conn, if_exists='append', chunksize=None, index=False)
    # print('ETH data appended')