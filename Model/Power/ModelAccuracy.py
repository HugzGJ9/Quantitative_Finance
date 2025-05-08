from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from Logger.Logger import mylogger
from Model.Power.RESPowerGeneration_model import getModelPipe
from Asset_Modeling.Energy_Modeling.data.data import fetchGenerationHistoryData


def evaluate_model_accuracy(hist, pipes, country="FR", holdout_days: int = 7, isShow=False):
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
            X_test = test_hist.drop(columns=[col for col in ["WIND", "SR"] if col in test_hist.columns])
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
            y_true = holdout[model_name][target]['y_true'].clip(lower=1e-6)  # Avoid divide-by-zero
            y_pred = holdout[model_name][target]['y_pred']
            error_per[f'{target} - {model_name}'] = ((y_true - y_pred) / y_true) * 100

            holdout[model_name][target].plot(title=f"{model_name} - {target}")
            if isShow:
                plt.show()

    SR_cols = [col for col in error_per.columns if "SR" in col]
    if isShow and SR_cols:
        plt.figure(figsize=(10, 5))
        error_per[SR_cols].plot(grid=True)
        plt.title("Solar (SR) Forecast Errors [%]")
        plt.ylabel("Relative Error (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    WIND_cols = [col for col in error_per.columns if "WIND" in col]
    if isShow and WIND_cols:
        plt.figure(figsize=(10, 5))
        error_per[WIND_cols].plot(grid=True)
        plt.title("Wind (WIND) Forecast Errors [%]")
        plt.ylabel("Relative Error (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return metrics, holdout

if __name__ == '__main__':
    history = fetchGenerationHistoryData('FR')
    pipes = {
        # "model1": getModelPipe(model_name="model_RES_generation_LGBMR_cleaned_fs"),
        # "model1_ns": getModelPipe(model_name="model_RES_generation_LGBMR_cleaned_fs_ns"),
        #
        # "model2": getModelPipe(model_name="model_RES_generation_LGBMR_fs"),
        # "model2_ns": getModelPipe(model_name="model_RES_generation_LGBMR_fs_ns"),
        #
        # "model3": getModelPipe(model_name="model_RES_generation_LGBMR_old"),
        # "model1": getModelPipe(model_name="model_RES_generation_LGBMR_fs"),
        # "model1_ns": getModelPipe(model_name="model_RES_generation_LGBMR_fs_ns"),
        "LGBMR": getModelPipe(model_name="model_RES_generation_LGBMR"),
        "LGBMR_cleaned": getModelPipe(model_name="model_RES_generation_LGBMR_cleaned"),
        "LGBMR_cleaned_tuned": getModelPipe(model_name="model_RES_generation_LGBMR_cleaned_pt"),
    }
    evaluate_model_accuracy(history, pipes, isShow=True)
