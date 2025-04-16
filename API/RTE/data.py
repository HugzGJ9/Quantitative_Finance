import datetime

import requests

from API.RTE.OAuth2 import getToken
from API.RTE.OAuth2 import API
import pandas as pd
from Logger.Logger import mylogger
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
        df = df[(df.index > now + pd.Timedelta(days=1)) & (df.index < now + pd.Timedelta(days=2))]
    elif APIname == 'Actual Generation':
        df = pd.DataFrame(data['actual_generations_per_production_type'][0]['values'])
        df.index = df['start_date']
        df = df.drop(columns=['start_date', 'end_date', 'updated_date'])
    return df
def getAPIdata(APIname:str, logger=False)->pd.DataFrame:
    access_token = getToken(APIname)
    headers = {
        'Authorization': f'Bearer {access_token}',
        # 'Content-Type': 'application/json',
        # 'start_date': '2025-01-01T00:00:00+02:00',
        # 'end_date': '2025-04-12T22:00:00+02:00',

    }
    response = requests.get(API[APIname]["token_url"], headers=headers)
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

if __name__ == '__main__':
    # getAPIdata(APIname="Wholesale Market", logger=True)
    # getAPIdata(APIname="Actual Generation", logger=True)
    getAPIdata(APIname="Generation Forecast", logger=True)



