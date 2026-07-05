"""Result: what a run returns — predictions, settlements, and diagnostics.

The vectorbt lesson: a backtest should hand back a rich object you can
interrogate (``.stats()``, ``.score``, ``.rank_against(problem)``), not a dict
of floats. The leaderboard and rebase-hillclimb's ``val_score:`` line both
read from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import typing as t

import numpy as np
import pandas as pd

if t.TYPE_CHECKING:  # pragma: no cover
    from ..problems.problem import Problem


@dataclass
class Result:
    problem: str
    model: str
    objective: str
    lower_is_better: bool
    split: str
    mode: str
    predictions: pd.DataFrame          # (asof, target_time) × output columns
    settlements: pd.DataFrame          # index asof; target_start/end, n_scored, score
    analysis: t.Dict[str, dict] = _dc_field(default_factory=dict)
    aggregate: str = "mean"            # how origin scores combine (metric-defined)

    @property
    def score(self) -> float:
        """Overall objective score: per-origin scores pooled by the number of
        scored timestamps for mean-type metrics, summed for sum-type ones
        (e.g. trading revenue)."""
        s = self.settlements
        valid = s[s["score"].notna() & (s["n_scored"] > 0)]
        if valid.empty:
            return float("nan")
        if self.aggregate == "sum":
            return float(valid["score"].sum())
        return float(np.average(valid["score"], weights=valid["n_scored"]))

    @property
    def n_scored(self) -> int:
        return int(self.settlements["n_scored"].sum())

    @property
    def n_origins(self) -> int:
        return len(self.settlements)

    def stats(self) -> dict:
        s = self.settlements["score"].dropna()
        out = {
            "problem": self.problem,
            "model": self.model,
            "objective": self.objective,
            "split": self.split,
            "mode": self.mode,
            "score": self.score,
            "n_origins": self.n_origins,
            "n_scored": self.n_scored,
            "origin_score_min": float(s.min()) if len(s) else float("nan"),
            "origin_score_max": float(s.max()) if len(s) else float("nan"),
        }
        for name, payload in self.analysis.items():
            for key, value in payload.items():
                out[f"{name}.{key}"] = value
        return out

    def to_leaderboard_row(self) -> dict:
        return {k: v for k, v in self.stats().items()
                if isinstance(v, (str, int, float, bool))}

    def rank_against(self, problem: "Problem") -> t.Optional[int]:
        """Position this score would have taken on the problem's published
        leaderboard (None if the problem has no reference scores)."""
        return problem.rank_of(self.score)

    def __repr__(self):
        return (f"Result({self.model!r} on {self.problem!r} [{self.split}]: "
                f"{self.objective}={self.score:.4f} over {self.n_origins} origins)")
