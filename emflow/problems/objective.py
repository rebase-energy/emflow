"""Objective: *the* metric a problem ranks on, with a direction.

Metrics are pure functions (see :mod:`emflow.problems.metrics`); an Objective
is the one a leaderboard sorts by — it adds direction (``lower_is_better``)
and comparison helpers. This is what rebase-hillclimb's ``val_score:`` line
ultimately reports.
"""

from __future__ import annotations

from dataclasses import dataclass
import typing as t

from .metrics import (
    Metric,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PeakTimingError,
    PinballLoss,
    RootMeanSquaredError,
)

__all__ = [
    "Objective", "Metric", "MeanAbsoluteError", "MeanSquaredError",
    "RootMeanSquaredError", "MeanAbsolutePercentageError", "PinballLoss",
    "PeakTimingError",
]


@dataclass
class Objective:
    metric: Metric
    lower_is_better: bool = True

    @property
    def name(self) -> str:
        return self.metric.name

    def calculate(self, y_true, y_pred) -> float:
        return self.metric.calculate(y_true, y_pred)

    def elementwise(self, y_true, y_pred):
        return self.metric.elementwise(y_true, y_pred)

    def is_better(self, a: float, b: float) -> bool:
        """True if score ``a`` beats score ``b``."""
        return a < b if self.lower_is_better else a > b
