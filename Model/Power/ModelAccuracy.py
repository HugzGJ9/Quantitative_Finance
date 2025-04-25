from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from Logger.Logger import mylogger
from Model.Power.RESPowerGeneration_model import _add_time_features, getModelPipe, getGenerationModelData


def evaluate_model_accuracy(weather, res_generation, pipes, country="FR", holdout_days: int = 7, isShow=False):
    hist = _add_time_features(pd.concat([weather, res_generation], axis=1).dropna(subset=weather.columns))
    hist = hist.dropna()
    cutoff_ts = hist.index.max() - pd.Timedelta(days=holdout_days)
    test_hist = hist[hist.index >= cutoff_ts]

    mylogger.logger.info(
        "Hold‑out: %s → %s (%d rows)",
        test_hist.index.min(),
        test_hist.index.max(),
        len(test_hist),
    )

    metrics = {}
    holdout = {}
    error_per = pd.DataFrame()
    for model_name, pipe in pipes.items():
        metrics[model_name] = {}
        holdout[model_name] = {}

        for target in ["WIND", "SR"]:
            if target not in test_hist.columns:
                mylogger.logger.warning(f"Target '{target}' not found in test data, skipping.")
                continue
            if target not in pipe:
                mylogger.logger.warning(f"Model '{model_name}' does not have a submodel for '{target}', skipping.")
                continue

            model = pipe[target]  # get the correct sub-model
            X_test = test_hist.drop(columns=[target])
            y_test = test_hist[target]
            y_pred = model.predict(X_test)

            metrics[model_name][target] = {
                "MAE": float(mean_absolute_error(y_test, y_pred)),
                "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
                "MAPE": float(mean_absolute_percentage_error(y_test, y_pred)),
                "R2": float(r2_score(y_test, y_pred)),
            }

            mylogger.logger.info(
                "%s — %s — MAE: %.3f, RMSE: %.3f, MAPE: %.2f%%, R²: %.3f",
                model_name,
                target,
                metrics[model_name][target]["MAE"],
                metrics[model_name][target]["RMSE"],
                metrics[model_name][target]["MAPE"] * 100,
                metrics[model_name][target]["R2"],
            )

            holdout[model_name][target] = pd.DataFrame(
                {"y_true": y_test, "y_pred": y_pred}, index=y_test.index
            )
            error_per[f'{target} - {model_name}'] =  (holdout[model_name][target]['y_true'] - holdout[model_name][target]['y_pred'])/ holdout[model_name][target]['y_pred'] * 100

            holdout[model_name][target].plot(title=f"{model_name} - {target}")
            if isShow:
                plt.show()

    SR_cols = [col for col in error_per.columns if "SR" in col]
    if isShow and SR_cols:
        error_per[SR_cols].plot()
        plt.title("Solar (SR) Forecast Errors")
        plt.legend()
        plt.show()

    WIND_cols = [col for col in error_per.columns if "WIND" in col]
    if isShow and WIND_cols:
        error_per[WIND_cols].plot()
        plt.title("Wind (WIND) Forecast Errors")
        plt.legend()
        plt.show()
    return metrics, holdout

if __name__ == '__main__':
    weather, res_generation = getGenerationModelData()
    pipes = {
        "model1": getModelPipe(model_name="model_RES_generation_LGBMR"),
        "model2": getModelPipe(model_name="model_RES_generation_XGBR"),
    }
    evaluate_model_accuracy(weather, res_generation, pipes, isShow=True)
