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
    for i in range(8):
        start = pd.Timestamp('20240201T0001', tz='Europe/Paris') - pd.DateOffset(years=i)
        end = pd.Timestamp('20250101T0001', tz='Europe/Paris') - pd.DateOffset(years=i)
        print(f"{start} - {end}")
        data = getPriceDaHist(start=start, end=end)
        data.index = data.index.tz_convert('UTC')
        updateDfSupabase(data, 'DAPowerPriceFR')

    # data = getWeatherData(cfg, 'history')
    # names = ['Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
    #          'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance', 'Cloud_Cover',
    #          'Cloud_Cover_Low', 'Cloud_Cover_Mid', 'Cloud_Cover_High',
    #          'Temperature_2m', 'Relative_Humidity_2m', 'Dew_Point_2m',
    #          'Precipitation', 'Wind_Speed_100m', 'Wind_Direction_100m',
    #          'Wind_Gusts_10m', 'Surface_Pressure']
    # data.columns = names
    # insertDfSupabase(data, 'WeatherFR')
    # updateDfSupabase(data, 'WeatherFR')
    # start = pd.Timestamp("2010-04-01", tz="Europe/Paris")
    # end = pd.Timestamp("2025-04-10", tz="Europe/Paris")
    # data = getInstalledCapacityData(start=start, end=end)
    # insertDfSupabase(data, 'InstalledCapacityFR')
    # df = getDfSupabase('DAPowerPriceFR')
    # df

    # removeAllRowsSupabase('GenerationFR')
