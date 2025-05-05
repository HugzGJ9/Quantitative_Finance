from __future__ import annotations
import os
from typing import Dict, List
import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Asset_Modeling.Energy_Modeling.data.data import fetchGenerationHistoryData
from Logger.Logger import mylogger
from API.OPENMETEO.Config_class import cfg
from xgboost import XGBRegressor
from Model.Power.dataProcessing import visualize_correlations, dataRESGenerationCleaning

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
def train(df: pd.DataFrame, TARGETS, techno: str, model_use="LGBMRegressor"):
    feats = TARGETS[techno]
    mask = df[feats + [techno]].notnull().all(axis=1)
    X, y = df.loc[mask, feats], df.loc[mask, techno]
    pipe = _build_pipe(feats, model=model_use)
    pipe.fit(X, y)
    return pipe

def builGenerationModel(hist, TARGETS, model_use="LGBMRegressor", country="FR", holdout_days:int=30, model_name='model_RES_generation') -> None:
    cutoff_ts = hist.index.max() - pd.Timedelta(days=holdout_days)
    train_hist = hist[hist.index < cutoff_ts]

    mylogger.logger.info(
        "Training on %s → %s (%d rows).",
        train_hist.index.min(),
        train_hist.index.max(),
        len(train_hist))

    models = {t: train(hist, TARGETS, t, model_use=model_use) for t in ("WIND", "SR")}
    joblib.dump(models, f"models_pkl/{model_name}.pkl")
    return

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

if __name__ == "__main__":
    history = fetchGenerationHistoryData('FR')
    # sr_features, wind_features = visualize_correlations(history, top_n=15)
    # sr_features = ['Solar_Radiation',
    #             'Diffuse_Radiation',
    #             'hour_cos',
    #             'Direct_Normal_Irradiance',
    #             'is_day',
    #             'Relative_Humidity_2m',
    #             'Temperature_2m',
    #             'WIND_capa',
    #             'month_cos',
    #             'Wind_Speed_100m',
    #             'Dew_Point_2m',
    #             'Wind_Gusts_10m']
    # TARGETS: Dict[str, List[str]] = {
    #     "WIND": wind_features,
    #     "SR": sr_features,
    # }

    outlier_indices = set()
    outliers = dataRESGenerationCleaning(history, 'Solar_Radiation', 'SR', quantile_clip=0.9)
    outlier_indices.update(outliers.index.tolist())
    outliers = dataRESGenerationCleaning(history, 'Wind_Speed_100m', 'WIND', quantile_clip=0.9)
    outlier_indices.update(outliers.index.tolist())

    history_cleaned = history.drop(index=outlier_indices)

    # builGenerationModel(history_cleaned, TARGETS, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR_fs")
    # builGenerationModel(history_cleaned, TARGETS, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR_cleaned_fs_ns")
    #
    # # features = list(history.columns)
    # # features = [x for x in features if x not in ['create_at', 'SR', 'WIND']]
    # # TARGETS: Dict[str, List[str]] = {
    # #     "WIND": features,
    # #     "SR": features,
    # # }
    # #
    # # builGenerationModel(history, TARGETS, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR_features_all_std")
    # #
    TARGETS: Dict[str, List[str]] = {
        "WIND": ['Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
                 'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance', 'Cloud_Cover', 'Cloud_Cover_Low',
                 'Cloud_Cover_Mid', 'Cloud_Cover_High', 'Temperature_2m', 'Relative_Humidity_2m', 'Dew_Point_2m',
                 'Precipitation', 'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m', 'Surface_Pressure',
                 'is_day',
                 'month', 'hour', 'WIND_capa', 'SR_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'],
        "SR": ['Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
               'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance', 'Cloud_Cover', 'Cloud_Cover_Low',
               'Cloud_Cover_Mid', 'Cloud_Cover_High', 'Temperature_2m', 'Relative_Humidity_2m', 'Dew_Point_2m',
               'Precipitation', 'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m', 'Surface_Pressure',
               'is_day',
               'month', 'hour', 'WIND_capa', 'SR_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'],
    }
    # builGenerationModel(history, TARGETS, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR_old_ns")
    builGenerationModel(history_cleaned, TARGETS, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR_cleaned_old")

