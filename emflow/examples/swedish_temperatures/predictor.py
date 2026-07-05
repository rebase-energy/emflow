"""Least-squares autoregressive predictor for the Swedish temperature problem.

Fits an ordinary-least-squares AR model

    y_t = c + sum_i w_i * y_{t - lag_i}

on the training view and forecasts through the declarative feature path, which
gives it batch (vectorized-mode) execution for free. Implemented with
``numpy.linalg.lstsq`` only — no scikit-learn / statsmodels dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from emflow.features import Lag
from emflow.models.predictor import FeaturePredictor


class ARPredictor(FeaturePredictor):
    """OLS autoregression on hourly lags (default: previous day + weekly lag).

    Parameters
    ----------
    lags : iterable of int, optional
        Lag orders in hours. Defaults to ``1..24`` and ``168``.
    field : str
        Name of the target field to build lag features from.
    """

    def __init__(self, lags=None, field="temperature", name="LeastSquaresAR"):
        self.lags = sorted(lags) if lags is not None else list(range(1, 25)) + [168]
        self.field = field
        features = [Lag(field, tuple(f"{lag}h" for lag in self.lags))]
        super().__init__(features=features, name=name)
        self.intercept = 0.0
        self.weights = np.zeros(len(self.lags))

    def fit(self, train):
        """OLS on the training history: rows with any missing lag are dropped."""
        series = train.history(self.field).iloc[:, 0]
        X = np.column_stack([series.shift(lag).to_numpy() for lag in self.lags])
        y = series.to_numpy()
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() <= len(self.lags) + 1:
            raise ValueError(
                f"not enough complete training rows ({int(mask.sum())}) to fit "
                f"{len(self.lags)} lags"
            )
        A = np.column_stack([np.ones(mask.sum()), X[mask]])
        beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        self.intercept, self.weights = float(beta[0]), beta[1:]
        return self

    def predict_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        vals = np.full(len(X), np.nan)
        Xv = X.to_numpy(dtype=float)
        finite = np.isfinite(Xv).all(axis=1)
        # errstate guards against numpy's spurious matmul FPE warnings on
        # large (but fully finite) operands; the result is correct.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            vals[finite] = self.intercept + Xv[finite] @ self.weights
        return pd.DataFrame({"point": vals}, index=X.index)
