from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from API.OPENMETEO.Config_class import cfg
from API.OPENMETEO.data import getWeatherData
from Asset_Modeling.Energy_Modeling.data.data import fetchRESCapacityData, _add_time_features

from Model.Power.RESPowerGeneration_model import getModelPipe


def getWeatherForecastData(country='FR'):
    weather = getWeatherData(cfg, 'forecast')
    weather.index = weather.index.tz_convert('Europe/Paris')
    res_capacity = fetchRESCapacityData(country)
    res_capacity = res_capacity.rename(columns={'SR': 'SR_capa', 'WIND': 'WIND_capa'})
    weather['SR_capa'] = weather.index.year.map(res_capacity['SR_capa'])
    weather['WIND_capa'] = weather.index.year.map(res_capacity['WIND_capa'])
    return weather


def predictGeneration(models: Dict[str, Pipeline], fc: pd.DataFrame) -> pd.DataFrame:
    tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
    df = fc[fc.index.date == tomorrow]
    if df.empty: return pd.DataFrame()
    feat = _add_time_features(df)
    targets_solar = models['SR'].feature_names_in_
    targets_wind = models['WIND'].feature_names_in_
    wmask = feat[targets_wind].notnull().all(axis=1)
    feat.loc[wmask, "WIND"] = models["WIND"].predict(feat.loc[wmask, targets_wind])

    # dmask = (feat["is_day"] == 1) & feat[targets_solar].notnull().all(axis=1)
    dmask = feat[targets_solar].notnull().all(axis=1)
    feat.loc[dmask, "SR"] = models["SR"].predict(feat.loc[dmask, targets_solar])
    feat.loc[~dmask, "SR"] = 0.0
    feat["SR"] = feat["SR"].clip(lower=0)

    feat["total_generation_mw"] = feat["WIND"] + feat["SR"]
    return feat.reset_index()[["time", "WIND", "SR", "total_generation_mw"]]

def getGenerationForecastReport(model_name="model_RES_generation_LGBMR"):
    pipe = getModelPipe(model_name=model_name)
    fc = getWeatherForecastData()
    generation_forecast = predictGeneration(pipe, fc)
    # generation_forecast = generation_forecast.rename(columns={"SR": "SOLAR"})
    generation_forecast = generation_forecast.set_index('time')
    return generation_forecast

if __name__ == '__main__':
    pipe = getModelPipe(model_name="model_RES_generation_LGBMR2")
    fc = getWeatherForecastData()
    generation_forecast = predictGeneration(pipe, fc)
    print(generation_forecast)


