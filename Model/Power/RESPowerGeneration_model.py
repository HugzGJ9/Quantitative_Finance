from __future__ import annotations

import os
from typing import Dict, List

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Asset_Modeling.Energy_Modeling.data.data import fetchRESGenerationData, fetchRESCapacityData, fetchWeatherData
from Logger.Logger import mylogger
from API.OPENMETEO.Config_class import cfg
import numpy as np
from xgboost import XGBRegressor


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["is_day"] = (df_copy["Solar_Radiation"] > 2.0).astype(int)
    df_copy["month"] = df_copy.index.astype(str).str[5:7].astype(int)
    df_copy["hour"] = df_copy.index.astype(str).str[11:13].astype(int)
    try:
        df_copy['SR'] = df_copy.apply(lambda row: 0.0 if row["Solar_Radiation"] == 0 else row['SR'], axis=1)
    except KeyError:
        pass
    df_copy["month_sin"] = np.sin(2 * np.pi * df_copy.index.month / 12)
    df_copy["month_cos"] = np.cos(2 * np.pi * df_copy.index.month / 12)

    df_copy["hour_sin"] = np.sin(2 * np.pi * df_copy.index.hour / 24)
    df_copy["hour_cos"] = np.cos(2 * np.pi * df_copy.index.hour / 24)
    return df_copy

TARGETS: Dict[str, List[str]] = {
    "WIND": ['Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
             'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance', 'Cloud_Cover', 'Cloud_Cover_Low',
             'Cloud_Cover_Mid', 'Cloud_Cover_High', 'Temperature_2m', 'Relative_Humidity_2m', 'Dew_Point_2m',
            'Precipitation', 'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m', 'Surface_Pressure', 'is_day',
             'month', 'hour', 'WIND_capa', 'SR_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'],
    "SR": ['Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
             'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance', 'Cloud_Cover', 'Cloud_Cover_Low',
             'Cloud_Cover_Mid', 'Cloud_Cover_High', 'Temperature_2m', 'Relative_Humidity_2m', 'Dew_Point_2m',
            'Precipitation', 'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m', 'Surface_Pressure', 'is_day',
             'month', 'hour', 'WIND_capa', 'SR_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'],
}
def _build_pipe(feats: List[str], model="LGBMRegressor") -> Pipeline:
    if model=="LGBMRegressor":
        return Pipeline([
            ("prep", ColumnTransformer([("num", StandardScaler(), feats)])),
            ("model", LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=64, subsample=0.8,
                                    colsample_bytree=0.8, random_state=cfg.random_seed))
        ])
    elif model=="TabPFNRegressor":
        return Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), feats)
            ])),
            ("model", TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu'))
        ])
    elif model=="XGBRegressor":
        return Pipeline([
            ("prep", ColumnTransformer([("num", StandardScaler(), feats)])),
            ("model", XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,  # depth ~ similar to num_leaves in LGBM
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=cfg.random_seed,
                tree_method="hist"  # optional: speeds up training for medium/large datasets
            ))
        ])
def train(df: pd.DataFrame, target: str, model_use="LGBMRegressor"):
    feats = TARGETS[target]
    mask = df[feats + [target]].notnull().all(axis=1)
    X, y = df.loc[mask, feats], df.loc[mask, target]
    pipe = _build_pipe(feats, model=model_use)
    pipe.fit(X, y)
    return pipe

def builGenerationModel(weather, res_generation, model_use="LGBMRegressor", country="FR", holdout_days:int=7, model_name='model_RES_generation') -> None:
    weather = weather[~weather.index.duplicated(keep='first')]
    res_generation = res_generation[~res_generation.index.duplicated(keep='first')]

    hist = _add_time_features(pd.concat([weather, res_generation], axis=1).dropna(subset=weather.columns))
    hist= hist.dropna()
    cutoff_ts = hist.index.max() - pd.Timedelta(days=holdout_days)
    train_hist = hist[hist.index < cutoff_ts]

    mylogger.logger.info(
        "Training on %s → %s (%d rows).",
        train_hist.index.min(),
        train_hist.index.max(),
        len(train_hist))

    models = {t: train(hist, t, model_use=model_use) for t in ("WIND", "SR")}
    joblib.dump(models, f"models_pkl/{model_name}.pkl")
    return
    # fc = fetch_weather_data(cfg, 'forecast')
    # tomorrow = predict_tomorrow(models, fc)
    # if tomorrow.empty:
    #     mylogger.logger.warning("No complete forecast rows for tomorrow."); return
    # avg = tomorrow["total_generation_mw"].mean()
    # mylogger.logger.info("Average RES generation on %s: %.1f MW", tomorrow.time.dt.date.iloc[0], avg)
    # print(tomorrow.head(24))

def getModelPipe(model_name="model_RES_generation"):
    current_path = os.getcwd()
    while os.path.basename(current_path) != "Quantitative_Finance":
        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError("Project root 'Quantitative_Finance' not found.")
        current_path = parent

    model_path = os.path.join(current_path, "Model", "Power", "models_pkl", model_name + ".pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    return joblib.load(model_path)

def getGenerationModelData(country='FR'):
    weather = fetchWeatherData(country)
    res_generation = fetchRESGenerationData(country)
    res_capacity = fetchRESCapacityData(country)
    res_capacity = res_capacity.rename(columns={'SR': 'SR_capa', 'WIND': 'WIND_capa'})
    weather['SR_capa'] = weather.index.year.map(res_capacity['SR_capa'])
    weather['WIND_capa'] = weather.index.year.map(res_capacity['WIND_capa'])

    return weather, res_generation


if __name__ == "__main__":
    weather, res_generation = getGenerationModelData()
    builGenerationModel(weather, res_generation, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR")
    builGenerationModel(weather, res_generation, model_use="XGBRegressor", model_name="model_RES_generation_XGBR")
