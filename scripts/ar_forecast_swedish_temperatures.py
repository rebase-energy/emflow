"""Train and evaluate a least-squares AR temperature forecaster (Stockholm).

Loads the emflow problem ``swedish-temperatures:ar``, fits an autoregressive
model on all data before 2026, forecasts 2026 one hour ahead, and reports MAE /
RMSE against two naive baselines (persistence and seasonal-24h).

Run:
    python scripts/ar_forecast_swedish_temperatures.py
"""

from __future__ import annotations

import pandas as pd

import emflow as ef
from emflow.examples.swedish_temperatures.predictor import ARPredictor
from emflow.problems.objective import MeanAbsoluteError, MeanSquaredError

MAE = MeanAbsoluteError()
MSE = MeanSquaredError()


def scores(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """MAE and RMSE, nan-aware (pairwise), with predictions aligned to y_true."""
    y_pred = y_pred.reindex(y_true.index)
    return MAE.calculate(y_true, y_pred), MSE.calculate(y_true, y_pred, squared=False)


def main() -> None:
    dataset, env, objective = ef.load_problem("swedish-temperatures:ar")

    init, forecast_input = env.reset()
    train_target = init["target"]
    series = forecast_input.iloc[:, 0]  # full Stockholm series (Series)

    print(f"Station    : {dataset.collection.members[0].name}")
    print(f"Train range: {train_target.index.min()} -> {train_target.index.max()}  "
          f"({len(train_target):,} hours)")

    # --- Fit + 1-step-ahead AR forecast -------------------------------------
    model = ARPredictor()
    model.train(train_target)
    preds_full = model.predict(forecast_input)

    _, test_target, done = env.step()
    y = test_target.iloc[:, 0]
    ar_pred = preds_full.iloc[:, 0]

    print(f"Test range : {y.index.min()} -> {y.index.max()}  ({len(y):,} hours)")
    print(f"Lags       : {model.lags}")
    assert done and train_target.index.max() < env.test_start <= y.index.min()

    # --- Baselines on the same test set -------------------------------------
    persistence = series.shift(1)    # y_hat_t = y_{t-1}
    seasonal = series.shift(24)      # y_hat_t = y_{t-24}

    rows = {
        "AR (lstsq, lags 1-24+168)": scores(y, ar_pred),
        "Persistence (t-1)": scores(y, persistence),
        "Seasonal naive (t-24)": scores(y, seasonal),
    }
    n_eval = int(y.reindex(ar_pred.index).notna().sum())

    print(f"\nEvaluated on {n_eval:,} non-missing test hours\n")
    print(f"{'Model':<28}{'MAE (°C)':>12}{'RMSE (°C)':>12}")
    print("-" * 52)
    for name, (mae, rmse) in rows.items():
        print(f"{name:<28}{mae:>12.3f}{rmse:>12.3f}")

    # --- Coefficient plausibility -------------------------------------------
    intercept, weights = model.coefs[train_target.columns[0]]
    lag1_w = weights[model.lags.index(1)]
    print(f"\nParameters : {len(weights) + 1} (intercept + {len(weights)} lags)")
    print(f"Intercept  : {intercept:.4f}   lag-1 weight: {lag1_w:.4f} "
          f"(max |weight| at lag {model.lags[abs(weights).argmax()]})")

    ar_mae = rows["AR (lstsq, lags 1-24+168)"][0]
    if ar_mae < min(rows["Persistence (t-1)"][0], rows["Seasonal naive (t-24)"][0]):
        print("\nAR beats both naive baselines on MAE. ✓")
    else:
        print("\nWARNING: AR did not beat the naive baselines.")


if __name__ == "__main__":
    main()
