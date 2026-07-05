"""Analyzers: observers of a run, fed each settlement as it lands.

The backtrader idea — one evaluation loop, many pluggable observers. Analyzers
never influence the run; they accumulate diagnostics and report in
``Result.analysis``.
"""

from __future__ import annotations

from abc import ABC
import typing as t

import numpy as np
import pandas as pd

from ..problems.metrics import quantile_columns


class Analyzer(ABC):
    """Base analyzer. ``setup`` receives the run context; ``on_settlement``
    each scored origin; ``finalize`` returns the diagnostics dict."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def setup(self, feed, target_field, target_column, objective) -> None:
        self.feed = feed
        self.target_field = target_field
        self.target_column = target_column
        self.objective = objective

    def on_settlement(self, record) -> None:  # pragma: no cover - interface
        pass

    def finalize(self) -> dict:  # pragma: no cover - interface
        return {}


class PersistenceSkill(Analyzer):
    """Scores a persistence baseline (last knowable value at each origin,
    broadcast over its targets) and reports the model's skill against it.
    A model that can't beat this hasn't learned anything."""

    def __init__(self):
        self._baseline_scores: t.List[float] = []
        self._model_scores: t.List[float] = []
        self._weights: t.List[int] = []
        self._disabled = False

    def setup(self, feed, target_field, target_column, objective) -> None:
        super().setup(feed, target_field, target_column, objective)
        # Env-settled objectives (trading revenue) can't score a synthetic
        # baseline from (actuals, prediction) alone — opt out cleanly.
        self._disabled = getattr(objective.metric, "settled_by_env", False)

    def on_settlement(self, record) -> None:
        if self._disabled:
            return
        hist = self.feed.history(record.origin.asof, self.target_field)
        col = hist[record.origin.column or self.target_column].dropna()
        if col.empty:
            return
        # Match the submission's output shape: for quantile predictions the
        # baseline is the point-mass persistence (last value at every level).
        levels = quantile_columns(record.prediction) or ["point"]
        baseline = pd.DataFrame({lvl: col.iloc[-1] for lvl in levels},
                                index=record.origin.target_index)
        n = int((record.actuals.notna()).sum())
        if n == 0:
            return
        self._baseline_scores.append(self.objective.calculate(record.actuals, baseline))
        self._model_scores.append(record.score)
        self._weights.append(n)

    def finalize(self) -> dict:
        if not self._weights:
            return {}
        w = np.asarray(self._weights, dtype=float)
        base = float(np.average(self._baseline_scores, weights=w))
        model = float(np.average(self._model_scores, weights=w))
        beats = self.objective.is_better(model, base)
        out = {"persistence_score": base, "model_score": model, "beats_persistence": bool(beats)}
        if base != 0:
            out["skill"] = float(1.0 - model / base) if self.objective.lower_is_better \
                else float(model / base - 1.0)
        return out


class QuantileCalibration(Analyzer):
    """Empirical coverage per predicted quantile. A calibrated q10 should have
    ~10% of actuals below it; large gaps mean the intervals are lying."""

    def __init__(self):
        self._below: t.Dict[float, int] = {}
        self._total: t.Dict[float, int] = {}

    def on_settlement(self, record) -> None:
        levels = quantile_columns(record.prediction)
        if not levels:
            return
        y = record.actuals
        for q in levels:
            pred = record.prediction[q].reindex(y.index)
            ok = y.notna() & pred.notna()
            self._below[q] = self._below.get(q, 0) + int((y[ok] <= pred[ok]).sum())
            self._total[q] = self._total.get(q, 0) + int(ok.sum())

    def finalize(self) -> dict:
        out = {}
        for q in sorted(self._total):
            if self._total[q]:
                out[f"coverage_q{int(round(q * 100)):02d}"] = self._below[q] / self._total[q]
        return out


def default_analyzers(model=None) -> t.List[Analyzer]:
    analyzers: t.List[Analyzer] = [PersistenceSkill()]
    if model is None or getattr(model, "output_kind", "point") == "quantiles":
        analyzers.append(QuantileCalibration())
    return analyzers
