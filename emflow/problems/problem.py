"""Problem: the complete, self-describing benchmark bundle.

``load_problem("heftcom2024:forecasting")`` returns one of these. It carries
everything an :class:`~emflow.run.experiment.Experiment` needs — dataset,
environment factory, objective, schedule, splits — plus the published
reference scores that let a result be ranked against the historical field.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import typing as t

import pandas as pd

from ..data import Dataset
from .objective import Objective
from .schedule import IssueSchedule

if t.TYPE_CHECKING:  # pragma: no cover
    import gymnasium as gym


@dataclass(frozen=True)
class Splits:
    """Temporal splits, in *target time*.

    ``train_end``
        Training data is everything strictly before this timestamp.
    ``validation``
        Half-open ``[start, end)`` period whose origins are for model
        iteration — what agents hillclimb on.
    ``holdout``
        Half-open ``[start, end)`` period scored once by the Verifier. Never
        iterate against it.
    """

    train_end: pd.Timestamp
    validation: t.Tuple[pd.Timestamp, pd.Timestamp]
    holdout: t.Tuple[pd.Timestamp, pd.Timestamp]

    def __post_init__(self):
        object.__setattr__(self, "train_end", pd.Timestamp(self.train_end))
        for attr in ("validation", "holdout"):
            start, end = getattr(self, attr)
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            if end <= start:
                raise ValueError(f"{attr}: end must be after start")
            if start < self.train_end:
                raise ValueError(f"{attr} starts before train_end — splits must not overlap training data")
            object.__setattr__(self, attr, (start, end))

    def period(self, split: str) -> t.Tuple[pd.Timestamp, pd.Timestamp]:
        if split not in ("validation", "holdout"):
            raise ValueError(f"split must be 'validation' or 'holdout', got {split!r}")
        return getattr(self, split)


@dataclass(frozen=True)
class RefScore:
    """One row of a competition's published final leaderboard."""

    rank: int
    team: str
    score: float


@dataclass
class Problem:
    """A benchmark problem. Environments are created fresh per run via
    :meth:`env` — never share a live env between runs."""

    name: str
    dataset: t.Union[Dataset, str, t.Callable[[], Dataset]]
    make_env: t.Callable[["Problem", str], "gym.Env"]
    objective: Objective
    schedule: IssueSchedule
    splits: Splits
    description: t.Optional[str] = None
    reference_scores: t.List[RefScore] = _dc_field(default_factory=list)

    _dataset_cache: t.Optional[Dataset] = _dc_field(default=None, repr=False, compare=False)

    def load_dataset(self) -> Dataset:
        """Resolve the dataset (lazily, cached): a Dataset instance, an
        ``rb://`` manifest reference, or a zero-arg loader callable."""
        if self._dataset_cache is None:
            ds = self.dataset
            if isinstance(ds, str):
                ds = Dataset.from_manifest(ds)
            elif callable(ds) and not isinstance(ds, Dataset):
                ds = ds()
            if not isinstance(ds, Dataset):
                raise TypeError(f"problem {self.name!r}: dataset resolved to {type(ds).__name__}")
            self._dataset_cache = ds
        return self._dataset_cache

    def env(self, split: str = "validation") -> "gym.Env":
        """Build a fresh environment over the given split's origins."""
        return self.make_env(self, split)

    def origins(self, split: str = "validation"):
        start, end = self.splits.period(split)
        return self.schedule.origins(start, end)

    def rank_of(self, score: float) -> t.Optional[int]:
        """1-based position ``score`` would have taken on the published
        leaderboard (None if no reference scores)."""
        if not self.reference_scores:
            return None
        beaten = sum(1 for r in self.reference_scores
                     if self.objective.is_better(score, r.score))
        return len(self.reference_scores) - beaten + 1

    def __repr__(self):
        return f"Problem({self.name!r}, objective={self.objective.name})"
