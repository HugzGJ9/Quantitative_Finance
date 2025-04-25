import numpy as np
import pytz
from supabase import create_client
import pandas as pd

from API.OPENMETEO.Config_class import cfg
from API.OPENMETEO.data import getWeatherData
from Keypass.key_pass import API_SUPABASE
from API.ENTSOE.data import getGenerationData, getPriceDaHist, getInstalledCapacityData
from Logger.Logger import mylogger

def getAccessSupabase(App: str):
    if App in API_SUPABASE.keys():
        SUPABASE_URL = API_SUPABASE[App]['token_url']
        SUPABASE_SERVICE_KEY = API_SUPABASE[App]['key']
    else:
        mylogger.logger.error(f"APP : {App} not found in SUPABASE Database.")
        return
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return  supabase

def getDfSupabase(db_name):
    supabase = getAccessSupabase(db_name)
    df = pd.DataFrame(supabase.table(db_name).select("*").execute().data)
    return df
def removeAllRowsSupabase(db_name):
    supabase = getAccessSupabase(db_name)
    data = supabase.table(db_name).select("id").execute().data
    ids_to_delete = [item['id'] for item in data]
    for row_id in ids_to_delete:
        supabase.table(db_name).delete().eq('id', row_id).execute()
    print(f"Removed {len(ids_to_delete)} rows from '{db_name}'")

def fetchRESGenerationData(country="FR"):
    if country == "FR":
        df = getDfSupabase('GenerationFR')
    else:
        df = getDfSupabase('GenerationFR')
    df['id'] = pd.to_datetime(df['id'], utc=True)
    df.index = df['id']
    df.index.name = 'time'
    df = df[['SR', 'WIND']]
    res_capacity = fetchRESCapacityData(country)
    res_capacity = res_capacity.rename(columns={'SR': 'SR_capa', 'WIND': 'WIND_capa'})
    df['SR_capa'] = df.index.year.map(res_capacity['SR_capa'])
    df['WIND_capa'] = df.index.year.map(res_capacity['WIND_capa'])
    df = df[(df['SR'] <= df['SR_capa']) & (df['WIND'] <= df['WIND_capa'])]
    df = df.drop(columns=['SR_capa', 'WIND_capa'])
    df = df[df.index.year < 2025]
    return df
def fetchRESCapacityData(country="FR"):
    if country == "FR":
        df = getDfSupabase('InstalledCapacityFR')
    else:
        df = getDfSupabase('InstalledCapacityFR')
    df['id'] = pd.to_datetime(df['id'], utc=True)
    df.index = df['id']
    df.index = df.index.year
    df.index.name = 'time'
    df['WIND'] = df['WOF'] + df['WON']
    df = df[['Solar', 'WIND']]
    df = df.rename(columns={'Solar': 'SR'})

    return df

def fetchWeatherData(country="FR"):
    if country == "FR":
        df = getDfSupabase('WeatherFR')
    else:
        df = getDfSupabase('WeatherFR')
    df['id'] = pd.to_datetime(df['id'], utc=True)
    df = df.set_index('id')
    df.index.name = 'time'
    return df
def insertDfSupabase(df, db_name):
    supabase = getAccessSupabase(db_name)
    df.index.name = 'id'
    data = df.reset_index()

    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    data = data.replace({np.nan: None, np.inf: None, -np.inf: None})
    data = data.where(pd.notnull(data), None)
    data = data.to_dict(orient="records")
    supabase.table(db_name).insert(data).execute()
def updateDfSupabase(df, db_name):
    supabase = getAccessSupabase(db_name)
    df.index.name = 'id'
    data = df.reset_index()

    # 1. Handle datetime formatting
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # 2. Replace NaN, NaT, inf, -inf by None
    data = data.replace({np.nan: None, np.inf: None, -np.inf: None})

    # 3. Confirm no weird types
    data = data.where(pd.notnull(data), None)

    # 4. Convert to dict
    data = data.to_dict(orient="records")

    # 5. Upsert
    supabase.table(db_name).upsert(data, ignore_duplicates=True).execute()


if __name__ == '__main__':
    for i in range(10*6):
        start = pd.Timestamp('20200301', tz='Europe/Paris') - pd.DateOffset(months=i*2)
        end = pd.Timestamp('20200601', tz='Europe/Paris') - pd.DateOffset(months=i*2)
        print(f"{start} - {end}")
        data = getGenerationData(start=start, end=end)
        data.index = data.index.tz_convert('UTC')
        updateDfSupabase(data, 'GenerationFR')

    # start = pd.Timestamp('20241201', tz='Europe/Paris')
    # end = pd.Timestamp('20250401', tz='Europe/Paris')
    # print(f"{start} - {end}")
    # data = getGenerationData(start=start, end=end)
    # data.index = data.index.tz_convert('UTC')
    # updateDfSupabase(data, 'GenerationFR')
    # removeAllRowsSupabase('GenerationFR')
