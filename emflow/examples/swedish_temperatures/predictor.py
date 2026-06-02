"""Least-squares autoregressive predictor for the Swedish temperature dataset.

Fits an ordinary-least-squares AR model independently for each column of the
training DataFrame:

    y_t = c + sum_i w_i * y_{t - lag_i}

and produces **1-step-ahead** forecasts using the *true* observed lagged values
(no recursive feedback of predictions). Implemented with ``numpy.linalg.lstsq``
only — no scikit-learn / statsmodels dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import emflow as ef


class ARPredictor(ef.Predictor):
    """OLS autoregression, fit per column, evaluated 1-step-ahead.

    Parameters
    ----------
    lags : iterable of int, optional
        Lag orders (in hours) used as regressors. Defaults to the previous full
        day plus the weekly lag: ``1..24`` and ``168``.
    name : str
        Model name.
    """

    def __init__(self, lags=None, name="LeastSquaresAR"):
        self.name = name
        self.lags = sorted(lags) if lags is not None else list(range(1, 25)) + [168]
        self.max_lag = max(self.lags)
        self.coefs = {}  # column -> (intercept: float, weights: ndarray) or None

    def _design_matrix(self, series: pd.Series) -> np.ndarray:
        """Build an (n_obs, n_lags) matrix of lagged values (with NaNs)."""
        return np.column_stack([series.shift(lag).to_numpy() for lag in self.lags])

    def train(self, train_df):
        """Fit one AR model per column via least squares.

        Rows with any NaN in the target or its required lags are dropped before
        fitting. A column with too few complete observations is left unfit
        (stored as ``None``) and will predict NaN.
        """
        if isinstance(train_df, pd.Series):
            train_df = train_df.to_frame()

        self.coefs = {}
        for col in train_df.columns:
            series = train_df[col]
            X = self._design_matrix(series)
            y = series.to_numpy()
            mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if mask.sum() <= len(self.lags) + 1:
                self.coefs[col] = None
                continue
            X_fit = np.column_stack([np.ones(mask.sum()), X[mask]])
            beta, *_ = np.linalg.lstsq(X_fit, y[mask], rcond=None)
            self.coefs[col] = (float(beta[0]), beta[1:])
        return self

    def predict(self, input):
        """1-step-ahead forecasts for every timestamp in ``input``.

        ``input`` must contain enough leading history for the lags to resolve
        (e.g. the test period preceded by ``max_lag`` hours, or the full series).
        Predictions are NaN where a required lag is missing or the column was not
        fit. The returned DataFrame shares ``input``'s index and columns; align it
        to the evaluation period with ``.reindex(test_target.index)``.
        """
        if isinstance(input, pd.Series):
            input = input.to_frame()

        preds = {}
        for col in input.columns:
            series = input[col]
            coef = self.coefs.get(col)
            if coef is None:
                preds[col] = np.full(len(series), np.nan)
                continue
            intercept, weights = coef
            X = self._design_matrix(series)
            # Only score rows whose lags are all present; leave the rest NaN
            # (avoids propagating NaN through the matmul).
            out = np.full(len(series), np.nan)
            finite = np.isfinite(X).all(axis=1)
            # errstate guards against numpy's spurious matmul FPE warnings on
            # large (but fully finite) operands; the result is correct.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                out[finite] = intercept + X[finite] @ weights
            preds[col] = out

        return pd.DataFrame(preds, index=input.index, columns=input.columns)
