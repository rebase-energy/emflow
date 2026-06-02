"""Example submission — a reference for how a model submission should look.

This is a *complete, self-contained* example for the ``swedish-temperatures:ar``
problem. Copy it, rename ``ExamplePredictor`` to your own model, and replace the
``train`` / ``predict`` bodies. The only thing the verifier needs is a module-level
``emflow.Predictor`` instance:

    model = ExamplePredictor()      # a FRESH, UNTRAINED model

The verifier (``scripts/verify_submission.py``) will:
  1. pick up your ``model`` (untrained),
  2. train it on the official pre-2026 split via ``model.train(train_df)``,
  3. score it with **strict walk-forward** on the held-out 2026 hours.

(If you prefer, you can instead expose a ``get_model() -> emflow.Predictor``
factory — the verifier accepts either form.)

You never see the test targets and you cannot train on them, so the score is
trustworthy. Self-check before submitting::

    python scripts/verify_submission.py submissions/example_submission.py

----------------------------------------------------------------------------
The model
----------------------------------------------------------------------------
A least-squares autoregression on the most recent hours:

    y_t = c + sum_i w_i * y_{t-lag_i}        (lags = 1, 2, 3 hours)

Fit per column with ``numpy.linalg.lstsq`` — no scikit-learn / statsmodels
needed. Short lags are deliberately strong for *1-step-ahead* hourly
temperature, and this comfortably beats the persistence and seasonal-naive
baselines.

Want to do better? This is where the intern earns their keep — e.g. add the
daily/weekly lags (24, 168), a per-hour-of-day or per-month offset, exogenous
NWP features, or swap the OLS for a gradient-boosted / neural model. Keep the
same ``train`` / ``predict`` contract and the verifier scores it the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import emflow as ef


class ExamplePredictor(ef.Predictor):
    """OLS autoregression on recent lags, fit independently per column.

    Parameters
    ----------
    lags : iterable of int
        Lag orders (hours) used as regressors. Default: 1, 2, 3.
    name : str
        Submission name shown on the scorecard / leaderboard.
    """

    def __init__(self, lags=(1, 2, 3), name="example-ar"):
        # IMPORTANT: set self.name so the scorecard / leaderboard can label you.
        self.name = name
        self.lags = sorted(lags)
        self.max_lag = max(self.lags)
        # column -> (intercept: float, weights: ndarray) or None if unfit
        self.coefs: dict = {}

    # -- feature construction -------------------------------------------------
    def _design_matrix(self, series: pd.Series) -> np.ndarray:
        """Build an (n_obs, n_lags) matrix of lagged values.

        Row ``t`` holds the values at ``t-lag`` for each lag — i.e. only the
        past, never the value at ``t`` itself. That is what keeps the model
        honest under walk-forward evaluation.
        """
        return np.column_stack([series.shift(lag).to_numpy() for lag in self.lags])

    # -- training -------------------------------------------------------------
    def train(self, train_df):
        """Fit one AR model per column via least squares on complete rows only.

        Rows with any NaN in the target or its required lags are dropped before
        fitting. A column with too little data is left unfit (predicts NaN).
        """
        if isinstance(train_df, pd.Series):
            train_df = train_df.to_frame()

        self.coefs = {}
        for col in train_df.columns:
            series = train_df[col]
            X = self._design_matrix(series)
            y = series.to_numpy()
            mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if mask.sum() <= len(self.lags) + 1:  # need more rows than params
                self.coefs[col] = None
                continue
            X_fit = np.column_stack([np.ones(mask.sum()), X[mask]])
            beta, *_ = np.linalg.lstsq(X_fit, y[mask], rcond=None)
            self.coefs[col] = (float(beta[0]), beta[1:])
        return self

    # -- prediction -----------------------------------------------------------
    def predict(self, input):
        """1-step-ahead forecast for every timestamp in ``input``.

        Returns a DataFrame sharing ``input``'s index/columns. The verifier reads
        the value at the LAST timestamp of ``input`` (whose actual is hidden as
        NaN) as your forecast for that hour. Rows whose lags are missing — or
        unfit columns — stay NaN.
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
            out = np.full(len(series), np.nan)
            finite = np.isfinite(X).all(axis=1)  # only score rows with all lags present
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                out[finite] = intercept + X[finite] @ weights
            preds[col] = out

        return pd.DataFrame(preds, index=input.index, columns=input.columns)


# The submission: a FRESH, UNTRAINED predictor for the verifier to train + score.
model = ExamplePredictor()
