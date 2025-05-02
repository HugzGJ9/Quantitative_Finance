import datetime

import requests

from API.RTE.OAuth2 import getToken
from Keypass.key_pass import API_RTE
import pandas as pd
from Logger.Logger import mylogger
from datetime import datetime, timedelta
import pytz
import requests
timezone = pytz.timezone('Europe/Paris')
now = datetime.now(timezone)
today = now.replace(hour=0, minute=0, second=0, microsecond=0)
yesterday = today - timedelta(days=1)
today = today + timedelta(days=1)
# Correct format (ISO 8601)
today = today.isoformat(timespec='seconds')
yesterday = yesterday.isoformat(timespec='seconds')
def dataformating(APIname, data):
    if APIname == 'Wholesale Market':
        df = pd.DataFrame(data['france_power_exchanges'][0]['values'])
        df['date'] = df['start_date'].str[:10]
        df['time'] = df['start_date'].str[11:19]
        df = df.drop(columns=['start_date', 'end_date'])
        df = df[['date', 'time', 'value', 'price']]
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('datetime')
    elif APIname == 'Generation Forecast':
        forecasts = data['forecasts']  # replace with your actual variable if different
        records = []
        for forecast in forecasts:
            prod_type = forecast['production_type']
            for v in forecast['values']:
                timestamp = v['start_date']
                value = v['value']
                records.append({'timestamp': timestamp, prod_type: value})
        df = pd.DataFrame(records)
        df = df.groupby('timestamp').first().reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df.resample('H').mean()

        df['WIND'] = df['WIND_ONSHORE'] + df['WIND_OFFSHORE']
        now = pd.Timestamp.now(tz='Europe/Paris').normalize()
        df = df[(df.index > now+ pd.Timedelta(days=1)) & (df.index < now + pd.Timedelta(days=2))]
    elif APIname == 'Actual Generation':
        production = data['actual_generations_per_production_type']  # replace with your actual variable if different
        records = []
        for prod in production:
            prod_type = prod['production_type']
            for v in prod['values']:
                timestamp = v['start_date']
                value = v['value']
                records.append({'timestamp': timestamp, prod_type: value})
        df = pd.DataFrame(records)
        df = df.groupby('timestamp').first().reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df.resample('H').mean()
        df['WIND'] = df['WIND_ONSHORE'] + df['WIND_OFFSHORE']
    elif APIname == 'Actual Generation 15min':
        production = data['generation_mix_15min_time_scale']  # replace with your actual variable if different
        records = []
        for prod in production:
            prod_type = prod['production_type']
            for v in prod['values']:
                timestamp = v['start_date']
                value = v['value']
                records.append({'timestamp': timestamp, prod_type: value})
        df = pd.DataFrame(records)
        df = df.groupby('timestamp').first().reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df.rename(columns={'SOLAR':'SR'})
    return df
def getAPIdata(APIname:str, logger=False)->pd.DataFrame:
    access_token = getToken(APIname)
    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    # Query parameters (dates go here)
    params = {
        'start_date': yesterday,
        'end_date': today
    }
    if APIname =='Actual Generation 15min':
        response = requests.get(API_RTE[APIname]["token_url"], headers=headers, params=params)
    else:
        response = requests.get(API_RTE[APIname]["token_url"], headers=headers)

    if logger:
        mylogger.logger.info(f"Status code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        data = dataformating(APIname, data)
    else:
        if logger:
            mylogger.logger.error("Error response:")
            mylogger.logger.error(response.text)
    return data

def getAuctionDaData():
    df = getAPIdata(APIname="Wholesale Market", logger=True)
    df = df.set_index('datetime')
    df = df[['value', 'price']]
    df = df.rename(columns={'value': 'volume'})
    df.index = df.index.tz_localize('Europe/Paris')
    return df

if __name__ == '__main__':
    # df = getAPIdata(APIname="Wholesale Market", logger=True)
    df = getAPIdata(APIname="Actual Generation 15min", logger=True)
    # df = getAPIdata(APIname="Generation Forecast", logger=True)
    df



