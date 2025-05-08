from __future__ import annotations
from typing import List
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Asset_Modeling.Energy_Modeling.data.data import fetchGenerationHistoryData
from Logger.Logger import mylogger
from API.OPENMETEO.Config_class import cfg
from Model.Power.dataProcessing import dataRESGenerationCleaning
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scikeras.wrappers import KerasRegressor

from Model.Power.nn_architectures import _build_ffnn, _build_ffnn2
from Model.Power.targets import TARGETS

def _build_pipe(feats: List[str], model="LGBMRegressor") -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), feats)
    ])

    if model == "LGBMRegressor":
        model_obj = LGBMRegressor(
            n_estimators=999,
            learning_rate=0.1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.random_seed
        )
    elif model == "XGBRegressor":
        model_obj = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.random_seed,
            tree_method="hist"
        )
    elif model == "DNNRegressor":
        model_obj = KerasRegressor(
            model=_build_ffnn,
            model__input_dim=len(feats),
            epochs=1000,
            batch_size=32,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown model type: {model}")

    return Pipeline([
        ("prep", preprocessor),
        ("model", model_obj)
    ])
def train(df: pd.DataFrame, TARGETS, techno: str, model_use="LGBMRegressor"):
    feats = TARGETS[techno]
    mask = df[feats + [techno]].notnull().all(axis=1)
    X, y = df.loc[mask, feats], df.loc[mask, techno]
    pipe = _build_pipe(feats, model=model_use)

    pipe.fit(X, y)
    return pipe


def buildGenerationModel(hist, TARGETS, model_use="LGBMRegressor", country="FR", holdout_days: int = 30,
                         model_name='model_RES_generation') -> None:


    cutoff_ts = hist.index.max() - pd.Timedelta(days=holdout_days)
    train_hist = hist[hist.index < cutoff_ts]

    mylogger.logger.info(
        "Training on %s → %s (%d rows).",
        train_hist.index.min(),
        train_hist.index.max(),
        len(train_hist)
    )

    pipes = {t: train(hist, TARGETS, t, model_use=model_use) for t in ("WIND", "SR")}

    if model_use == "DNNRegressor":
        for techno, pipe in pipes.items():
            keras_model = pipe.named_steps["model"].model_

            keras_model.save_weights(f"models_pkl/{model_name}_{techno}_pipe_weights.h5")

            pipe.named_steps["model"].model_ = None

    joblib.dump(pipes, f"models_pkl/{model_name}.pkl", compress=0)

def getModelPipe(model_name="model_RES_generation_DNNR_cleaned"):
    import os
    import joblib
    import numpy as np

    current_path = os.getcwd()
    while os.path.basename(current_path) != "Quantitative_Finance":
        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError("Project root 'Quantitative_Finance' not found.")
        current_path = parent

    model_dir = os.path.join(current_path, "Model", "Power", "models_pkl")
    model_path = os.path.join(model_dir, model_name + ".pkl")

    pipes = joblib.load(model_path)

    if "DNN" in model_name:
        for techno, pipe in pipes.items():
            keras_wrapper = pipe.named_steps["model"]

            # 🔧 Rebuild model using dummy fit (triggers .model_ creation)
            n_features = len(pipe.named_steps["prep"].transformers[0][2])
            X_dummy = np.zeros((1, n_features))
            y_dummy = np.zeros((1,))
            keras_wrapper.fit(X_dummy, y_dummy)

            # ✅ Now it's safe to load weights
            weights_path = os.path.join(model_dir, f"{model_name}_{techno}_pipe_weights.h5")
            keras_wrapper.model_.load_weights(weights_path)

    return pipes

if __name__ == "__main__":
    history = fetchGenerationHistoryData('FR')
    # sr_features, wind_features = visualize_correlations(history, top_n=15)
    # TARGETS: Dict[str, List[str]] = {
    #     "WIND": wind_features,
    #     "SR": sr_features,
    # }

    outlier_indices = set()
    outliers = dataRESGenerationCleaning(history, 'Solar_Radiation', 'SR', quantile_clip=0.9)
    outlier_indices.update(outliers.index.tolist())
    outliers = dataRESGenerationCleaning(history, 'Wind_Speed_100m', 'WIND', quantile_clip=0.9)
    outlier_indices.update(outliers.index.tolist())
    # outliers = detect_multivariate_outliers_isoforest(history)
    # outlier_indices.update(outliers.index.tolist())


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
    # builGenerationModel(history, TARGETS, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR_old_ns")
    buildGenerationModel(history_cleaned, TARGETS, model_use="LGBMRegressor", model_name="model_RES_generation_LGBMR_cleaned_pt")
    # buildGenerationModel(history_cleaned, TARGETS, model_use="XGBRegressor", model_name="model_RES_generation_XGBR_cleaned")
    # buildGenerationModel(history_cleaned, TARGETS, model_use="DNNRegressor", model_name="model_RES_generation_DNNR_cleaned")

