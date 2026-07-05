"""Example submission — a complete, self-contained model for
``swedish-temperatures:ar``.

Copy it, rename ``ExamplePredictor``, and replace the ``fit`` / ``predict``
bodies. The verifier accepts either a module-level ``model = ...`` instance or
a ``get_model()`` factory; both must be FRESH and UNTRAINED — the verifier does
the training itself on the official split.

How evaluation works
--------------------
* ``fit(train)`` receives a **TimeView** frozen at the training cutoff:
  ``train.history("temperature")`` is the hourly Stockholm series up to there.
* ``predict(obs)`` receives an **Observation** per forecast origin:
  ``obs.history("temperature")`` is everything knowable at that origin, and
  ``obs.target_index`` is the hour you must forecast. Return a DataFrame with a
  ``"point"`` column on that index. The actual at the target hour is *not in*
  ``obs`` — peeking is structurally impossible.

Self-check::

    python scripts/verify_submission.py submissions/example_submission.py

The model: OLS autoregression on the last few hours,

    y_t = c + sum_i w_i * y_{t-lag_i}        (lags = 1, 2, 3 hours)

fit with ``numpy.linalg.lstsq`` — no sklearn needed. Short lags are strong for
1-step-ahead hourly temperature and comfortably beat persistence. Want to do
better? Add daily/weekly lags (24, 168), calendar features, or swap the OLS
for a gradient-boosted model — see ``emflow.FeaturePredictor`` for the
declarative-feature path that also gets you fast vectorized evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import emflow as ef

FIELD = "temperature"


class ExamplePredictor(ef.Predictor):
    """OLS autoregression on recent hourly lags."""

    def __init__(self, lags=(1, 2, 3), name="example-ar"):
        super().__init__(name=name)
        self.lags = sorted(lags)
        self.intercept = 0.0
        self.weights = np.zeros(len(self.lags))

    def fit(self, train):
        """Least squares on complete rows of the training history."""
        series = train.history(FIELD).iloc[:, 0]
        X = np.column_stack([series.shift(lag).to_numpy() for lag in self.lags])
        y = series.to_numpy()
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        A = np.column_stack([np.ones(mask.sum()), X[mask]])
        beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        self.intercept, self.weights = float(beta[0]), beta[1:]
        return self

    def predict(self, obs):
        """Forecast obs.target_index from the lags available at the origin."""
        series = obs.history(FIELD, window="2D").iloc[:, 0]
        preds = []
        for t in obs.target_index:
            lag_values = np.array([series.get(t - pd.Timedelta(hours=lag), np.nan)
                                   for lag in self.lags])
            preds.append(self.intercept + lag_values @ self.weights
                         if np.isfinite(lag_values).all() else np.nan)
        return pd.DataFrame({"point": preds}, index=obs.target_index)


# The submission: a FRESH, UNTRAINED predictor for the verifier to train + score.
model = ExamplePredictor()
