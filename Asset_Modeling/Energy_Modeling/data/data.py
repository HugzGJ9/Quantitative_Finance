from __future__ import annotations

import numpy as np
import pandas as pd

from API.SUPABASE.client import getDfSupabase


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
    # df = df[df.index.year < 2025]
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

def fetchRESGenerationMonthlyData(country="FR"):
    res_generation = fetchRESGenerationData(country)
    res_capacity = fetchRESCapacityData(country=country)
    res_capacity = res_capacity.rename(columns={'SR': 'SR_capa', 'WIND': 'WIND_capa'})
    res_generation['SR_capa'] = res_generation.index.year.map(res_capacity['SR_capa'])
    res_generation['WIND_capa'] = res_generation.index.year.map(res_capacity['WIND_capa'])

    max_SR_capa = res_generation['SR_capa'].max()
    max_WIND_capa = res_generation['WIND_capa'].max()

    res_generation['SR_normalized'] = res_generation['SR'] * (max_SR_capa / res_generation['SR_capa'])
    res_generation['WIND_normalized'] = res_generation['WIND'] * (max_WIND_capa / res_generation['WIND_capa'])

    res_generation_day = res_generation.resample('D').sum()
    res_generation_month = res_generation_day.resample('M').mean()
    if res_generation_month.index.tz is not None:
        res_generation_month.index = res_generation_month.index.tz_convert(None)
        res_generation_day.index = res_generation_day.index.tz_convert(None)
    res_generation_month.index = res_generation_month.index.to_period('M').to_timestamp()
    res_generation_day.index = res_generation_day.index.to_period('D').to_timestamp()

    res_generation_month = res_generation_month.drop(columns=['SR_capa', 'WIND_capa'])
    res_generation_day = res_generation_day.drop(columns=['SR_capa', 'WIND_capa'])

    return res_generation_month, res_generation_day
if __name__ == '__main__':
    df = fetchRESGenerationMonthlyData("FR")


def getGenerationModelData(country='FR'):
    weather = fetchWeatherData(country)
    res_generation = fetchRESGenerationData(country)
    res_capacity = fetchRESCapacityData(country)
    res_capacity = res_capacity.rename(columns={'SR': 'SR_capa', 'WIND': 'WIND_capa'})
    weather['SR_capa'] = weather.index.year.map(res_capacity['SR_capa'])
    weather['WIND_capa'] = weather.index.year.map(res_capacity['WIND_capa'])

    return weather, res_generation


def fetchGenerationHistoryData(country='FR'):
    weather, res_generation = getGenerationModelData(country)
    weather = weather[~weather.index.duplicated(keep='first')]
    res_generation = res_generation[~res_generation.index.duplicated(keep='first')]

    hist = _add_time_features(pd.concat([weather, res_generation], axis=1).dropna(subset=weather.columns))
    hist = hist.dropna()
    max_SR_capa = hist['SR_capa'].max()
    max_WIND_capa = hist['WIND_capa'].max()
    hist['SR'] = hist['SR'] * (max_SR_capa / hist['SR_capa'])
    hist['WIND'] = hist['WIND'] * (max_WIND_capa / hist['WIND_capa'])
    # hist = hist.drop(columns=['SR_capa', 'WIND_capa'])
    return hist


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["is_day"] = (df_copy["Solar_Radiation"] > 2.0).astype(int)
    df_copy["month"] = df_copy.index.astype(str).str[5:7].astype(int)
    df_copy["hour"] = df_copy.index.astype(str).str[11:13].astype(int)
    try:
        df_copy['SR'] = df_copy.apply(lambda row: 0.0 if row["Solar_Radiation"] < 3.0 else row['SR'], axis=1)
    except KeyError:
        pass
    df_copy["month_sin"] = np.sin(2 * np.pi * df_copy.index.month / 12)
    df_copy["month_cos"] = np.cos(2 * np.pi * df_copy.index.month / 12)

    df_copy["hour_sin"] = np.sin(2 * np.pi * df_copy.index.hour / 24)
    df_copy["hour_cos"] = np.cos(2 * np.pi * df_copy.index.hour / 24)
    return df_copy
