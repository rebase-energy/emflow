"""Predictor: the forecasting model contract.

A predictor is fit on a training :class:`~emflow.data.feed.TimeView` and
asked to forecast at :class:`~emflow.envs.forecast.Observation` s. Both are
feed-served, so a predictor physically cannot see past the information set —
there is nothing else to look at.

Contract
--------
``fit(train)``
    ``train`` is a TimeView frozen at the problem's training cutoff. Pull what
    you need via ``train.history(...)`` / ``train.forecasts(...)``.
``predict(obs)``
    Return a DataFrame indexed by ``obs.target_index`` with either a
    ``"point"`` column or float quantile columns matching ``self.quantiles``.

Set ``output_kind = "quantiles"`` and ``quantiles = (0.1, ..., 0.9)`` for
probabilistic models; the environment validates submissions against it.

:class:`FeaturePredictor` adds the declarative-feature path: declare
``features`` specs and implement ``predict_tabular(X)``; event-mode
``predict()`` then comes for free, and the model gains ``supports_batch``,
unlocking the vectorized execution mode (all origins in one call).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import typing as t

import pandas as pd

from .model import Model


class Predictor(Model, ABC):
    output_kind: str = "point"                       # "point" | "quantiles"
    quantiles: t.Optional[t.Tuple[float, ...]] = None
    supports_batch: bool = False

    def __init__(self, name: t.Optional[str] = None):
        super().__init__(name or type(self).__name__)

    def bind(self, problem) -> "Predictor":
        """Called by the Experiment before ``fit`` with the Problem being run.

        Gives models that need problem context (the schedule, to enumerate
        training origins; the dataset, for supervised feature frames) a
        sanctioned way to get it. Default just stores it as ``self.problem``.
        """
        self.problem = problem
        return self

    def fit(self, train) -> "Predictor":
        """Fit on a training TimeView. Optional — override for trainable models."""
        return self

    @abstractmethod
    def predict(self, obs) -> pd.DataFrame:
        """Forecast one origin: a DataFrame indexed by ``obs.target_index``."""

    def copy(self, name: t.Optional[str] = None) -> "Predictor":
        predictor = copy.deepcopy(self)
        if name is not None:
            predictor.name = name
        return predictor


class FeaturePredictor(Predictor):
    """Predictor over a declarative, point-in-time-correct feature matrix.

    Subclasses declare ``features`` (specs from :mod:`emflow.features`) and
    implement :meth:`predict_tabular`; they may also use
    :func:`emflow.features.materialize.supervised_frame` in ``fit``.
    """

    supports_batch = True
    features: t.Sequence = ()

    def __init__(self, features=None, name: t.Optional[str] = None):
        super().__init__(name)
        if features is not None:
            self.features = tuple(features)
        if not self.features:
            raise ValueError(f"{type(self).__name__} declares no feature specs")

    @abstractmethod
    def predict_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forecast from a feature matrix indexed by ``(asof, target_time)``.

        Must return a frame on the same index (or on ``target_time``) with the
        model's output columns. Must be row-wise pure: each row's forecast may
        depend only on that row's features — this is what makes single-origin
        and all-origins evaluation identical.
        """

    def predict(self, obs) -> pd.DataFrame:
        from ..features.materialize import materialize_observation

        X = materialize_observation(obs, self.features, obs.target_index)
        out = self.predict_tabular(X)
        if isinstance(out.index, pd.MultiIndex):
            out = out.droplevel("asof")
        return out
