"""Metrics: pure scoring functions with one uniform signature.

Every metric implements ``calculate(y_true, y_pred) -> float`` and
``elementwise(y_true, y_pred)``. The canonical prediction format everywhere in
emflow is a DataFrame indexed by target time whose columns are either

* ``["point"]`` — a point forecast, or
* quantile levels as floats (``0.1 ... 0.9``) — a probabilistic forecast.

``y_true`` is a Series (or single-column DataFrame) on the same index. NaNs are
ignored pairwise. A metric is *not* what a problem ranks on — that's the
:class:`~emflow.problems.objective.Objective`, which wraps a metric with a
direction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t

import numpy as np
import pandas as pd

POINT_COL = "point"


def _to_series(y) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"expected a single target column, got {list(y.columns)}")
        return y.iloc[:, 0]
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y, dtype=float))


def _point(y_pred) -> pd.Series:
    """Extract the point forecast from the canonical prediction frame."""
    if isinstance(y_pred, pd.DataFrame):
        if POINT_COL in y_pred.columns:
            return y_pred[POINT_COL]
        if 0.5 in y_pred.columns:  # median of a quantile forecast
            return y_pred[0.5]
        if y_pred.shape[1] == 1:
            return y_pred.iloc[:, 0]
        raise ValueError(
            f"cannot extract a point forecast from columns {list(y_pred.columns)}"
        )
    return _to_series(y_pred)


def quantile_columns(y_pred: pd.DataFrame) -> t.List[float]:
    return sorted(c for c in y_pred.columns if isinstance(c, float) and 0.0 < c < 1.0)


class Metric(ABC):
    """Base class for metrics. Stateless; safe to share across problems."""

    #: How per-origin scores combine into an overall score: "mean" (pooled by
    #: scored timestamps — error metrics) or "sum" (revenue-type metrics).
    aggregate: str = "mean"

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def elementwise(self, y_true, y_pred):
        """Per-timestamp scores (NaN where either side is missing)."""

    def calculate(self, y_true, y_pred) -> float:
        errs = np.asarray(self.elementwise(y_true, y_pred), dtype=float)
        if not np.isfinite(errs).any():
            return float("nan")
        return float(np.nanmean(errs))


class MeanAbsoluteError(Metric):
    def elementwise(self, y_true, y_pred):
        y, p = _to_series(y_true), _point(y_pred)
        p = p.reindex(y.index)
        return np.abs(y.to_numpy(float) - p.to_numpy(float))


class MeanSquaredError(Metric):
    def __init__(self, squared: bool = True):
        self.squared = squared

    @property
    def name(self):
        return "MeanSquaredError" if self.squared else "RootMeanSquaredError"

    def elementwise(self, y_true, y_pred):
        y, p = _to_series(y_true), _point(y_pred)
        p = p.reindex(y.index)
        return (y.to_numpy(float) - p.to_numpy(float)) ** 2

    def calculate(self, y_true, y_pred) -> float:
        mse = super().calculate(y_true, y_pred)
        return mse if self.squared else float(np.sqrt(mse))


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self):
        super().__init__(squared=False)


class MeanAbsolutePercentageError(Metric):
    def elementwise(self, y_true, y_pred):
        y, p = _to_series(y_true), _point(y_pred)
        p = p.reindex(y.index)
        yv = y.to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.abs(yv - p.to_numpy(float)) / np.abs(yv)
        ape[~np.isfinite(ape)] = np.nan
        return 100.0 * ape


class PinballLoss(Metric):
    """Average pinball (quantile) loss over the prediction's quantile columns.

    ``y_pred`` must carry quantile levels as float column names. If the metric
    was constructed with explicit ``quantiles``, the prediction must provide
    exactly those columns (competition contract); otherwise whatever quantile
    columns are present are scored.
    """

    def __init__(self, quantiles: t.Optional[t.Sequence[float]] = None):
        if quantiles is not None:
            quantiles = [float(q) for q in quantiles]
            if any(q <= 0.0 or q >= 1.0 for q in quantiles):
                raise ValueError("quantiles must lie strictly between 0 and 1")
        self.quantiles = quantiles

    def _levels(self, y_pred: pd.DataFrame) -> t.List[float]:
        present = quantile_columns(y_pred)
        if self.quantiles is None:
            if not present:
                raise ValueError("prediction has no quantile columns to score")
            return present
        missing = [q for q in self.quantiles if q not in set(present)]
        if missing:
            raise ValueError(f"prediction is missing required quantiles {missing}")
        return list(self.quantiles)

    def elementwise(self, y_true, y_pred):
        """Per-timestamp pinball loss, averaged across quantiles."""
        y = _to_series(y_true)
        levels = self._levels(y_pred)
        preds = y_pred[levels].reindex(y.index).to_numpy(float)
        yv = y.to_numpy(float)[:, None]
        q = np.asarray(levels)[None, :]
        err = yv - preds
        losses = np.where(err >= 0, q * err, (q - 1.0) * err)
        return np.nanmean(losses, axis=1)


class TradingRevenue(Metric):
    """Day-ahead trading revenue under HEFTCom24 settlement: per period,

        bid × DA_price + (actual − bid) × SS_price − λ·(actual − bid)²

    with λ = 0.07 — the exact formula reproduces the official archive's
    per-period revenues to 1e-10 (the quadratic term penalizes large
    imbalances).

    Needs market prices, which live on the environment's clock — so this metric
    is **settled by** :class:`~emflow.envs.trading.TradingEnv`, not computed
    from ``(y_true, y_pred)`` alone. It exists as a Metric for naming,
    direction and aggregation ("sum": competitions rank total revenue).
    """

    aggregate = "sum"
    settled_by_env = True

    def __init__(self, imbalance_penalty: float = 0.07):
        self.imbalance_penalty = imbalance_penalty

    def revenue(self, actuals, bids, da_price, ss_price):
        """Elementwise revenue given aligned series (the env calls this)."""
        imbalance = actuals - bids
        return (bids * da_price + imbalance * ss_price
                - self.imbalance_penalty * imbalance ** 2)

    def elementwise(self, y_true, y_pred):
        raise NotImplementedError(
            "TradingRevenue is settled by TradingEnv with market prices; it "
            "cannot be computed from predictions and actuals alone"
        )


class PeakTimingError(Metric):
    """BigDEAL-2022-style peak-timing score: per day, the absolute distance (in
    hours) between the true and predicted peak hour of ``y``.

    Both series are grouped by calendar day on their index; days missing from
    either side are skipped.
    """

    def elementwise(self, y_true, y_pred):
        y, p = _to_series(y_true), _point(y_pred)
        p = p.reindex(y.index)
        frame = pd.DataFrame({"y": y, "p": p}).dropna()
        if frame.empty:
            return np.array([np.nan])
        days = frame.groupby(frame.index.normalize())
        errors = []
        for _, day in days:
            true_peak = day["y"].idxmax()
            pred_peak = day["p"].idxmax()
            errors.append(abs((true_peak - pred_peak).total_seconds()) / 3600.0)
        return np.asarray(errors, dtype=float)
