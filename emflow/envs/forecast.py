"""ForecastEnv: rolling-origin forecasting as a real gymnasium environment.

The action *is* the forecast: a DataFrame indexed by the current origin's
target times, with a ``"point"`` column or quantile-level float columns. The
observation is a :class:`Observation` — a :class:`~emflow.data.portal.TimeView`
frozen at the origin's ``asof`` plus the target index to forecast. The model
never touches the dataset; the observation is its whole world.

Rewards are **delayed to settlement**, as in reality: submitting a day-ahead
forecast tells you nothing — the score arrives once the actuals do. At each
step the env settles every previously submitted origin whose targets have
become knowable (per the target field's availability rule) and pays their
scores as the reward (negated when the objective is lower-is-better, so more
reward is always better). ``info["settled"]`` carries the full
:class:`SettlementRecord` s for analyzers.
"""

from __future__ import annotations

from dataclasses import dataclass
import typing as t

import gymnasium as gym
import pandas as pd

from ..data import DataPortal, TimeView
from ..data.field import Field
from ..problems.metrics import POINT_COL, quantile_columns
from ..problems.objective import Objective
from ..problems.schedule import Origin


class Observation(TimeView):
    """What a model sees at one forecast origin: point-in-time data access
    (via :class:`TimeView`) plus the target index it must forecast."""

    def __init__(self, portal: DataPortal, origin: Origin):
        super().__init__(portal, origin.asof)
        self.target_index = origin.target_index

    def __repr__(self):
        return (f"Observation(asof={self.asof}, targets={self.target_index[0]} → "
                f"{self.target_index[-1]}, n={len(self.target_index)})")


@dataclass
class SettlementRecord:
    """One origin scored against its (now available) actuals."""

    origin: Origin
    prediction: pd.DataFrame
    actuals: pd.Series
    score: float


class ForecastEnv(gym.Env):
    """Rolling-origin forecast evaluation over a fixed list of origins.

    Parameters
    ----------
    portal:
        The point-in-time data gate.
    origins:
        The origins to step through (from ``problem.origins(split)``).
    target_field:
        Name of the actual field being forecast.
    objective:
        Scores each settled origin (prediction vs actuals).
    target_column:
        Column of the target field to score (default: its only column).
    train_end:
        Clock time of the training view served in ``reset()``'s
        ``info["train"]``.
    quantiles:
        If set, actions must provide exactly these quantile columns;
        if None, actions must provide a ``"point"`` column.
    """

    metadata = {"render_modes": []}

    def __init__(self, portal: DataPortal, origins: t.Sequence[Origin],
                 target_field: str, objective: Objective, *,
                 target_column=None, train_end=None,
                 quantiles: t.Optional[t.Sequence[float]] = None):
        if not origins:
            raise ValueError("ForecastEnv needs at least one origin")
        self.portal = portal
        self.origins = list(origins)
        self.target_field = target_field
        self.objective = objective
        self.quantiles = [float(q) for q in quantiles] if quantiles is not None else None
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None

        field = portal.dataset.field(target_field)
        if target_column is None:
            if field.frame.shape[1] != 1:
                raise ValueError(
                    f"target field {target_field!r} has columns "
                    f"{list(field.frame.columns)}; pass target_column"
                )
            target_column = field.frame.columns[0]
        self.target_column = target_column
        self._target_lag = field.availability_lag
        # Everything a settlement needs must be knowable: subclasses that pull
        # extra fields (e.g. market prices) raise this to the max of the lags.
        self._settle_lag = field.availability_lag

        self._i = 0
        self._pending: t.List[t.Tuple[Origin, pd.DataFrame]] = []

    # -- gym API ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self._pending = []
        info = {}
        if self.train_end is not None:
            # Strict boundary: a value stamped exactly train_end is a scored
            # target of the first evaluation origin, never training data.
            info["train"] = self.portal.view(self.train_end, strict=True)
        return Observation(self.portal, self.origins[0]), info

    def step(self, action):
        origin = self.origins[self._i]
        self._pending.append((origin, self._validate(action, origin)))
        self._i += 1

        terminated = self._i >= len(self.origins)
        if terminated:
            # Final flush: the run is over, settle everything that will ever
            # be knowable (clock = last target's availability time).
            last_end = max(o.target_end for o, _ in self._pending)
            settle_asof = last_end + self._settle_lag
            obs = None
        else:
            settle_asof = self.origins[self._i].asof
            obs = Observation(self.portal, self.origins[self._i])

        settled = self._settle(settle_asof)
        reward = sum(-r.score if self.objective.lower_is_better else r.score
                     for r in settled if pd.notna(r.score))
        info = {"settled": settled}
        return obs, float(reward), terminated, False, info

    # -- internals -----------------------------------------------------------------

    def _validate(self, action, origin: Origin) -> pd.DataFrame:
        if not isinstance(action, pd.DataFrame):
            raise TypeError(
                f"action must be a DataFrame indexed by target time, got "
                f"{type(action).__name__}"
            )
        missing = origin.target_index.difference(action.index)
        if len(missing):
            raise ValueError(
                f"action is missing forecasts for {len(missing)} target times "
                f"(first: {missing[0]})"
            )
        action = action.reindex(origin.target_index)
        if self.quantiles is not None:
            have = set(quantile_columns(action))
            need = set(self.quantiles)
            if not need.issubset(have):
                raise ValueError(
                    f"action must provide quantile columns {sorted(need)}, "
                    f"got {sorted(have) or list(action.columns)}"
                )
        elif POINT_COL not in action.columns:
            if action.shape[1] == 1:
                action = action.set_axis([POINT_COL], axis=1)
            else:
                raise ValueError(
                    f"action must have a {POINT_COL!r} column, got {list(action.columns)}"
                )
        return action

    def _settle(self, asof) -> t.List[SettlementRecord]:
        """Score every pending origin whose full target window is knowable at
        ``asof``; keep the rest pending."""
        settled, still_pending = [], []
        for origin, prediction in self._pending:
            if origin.target_end + self._settle_lag <= asof:
                actuals = self.portal.actuals_between(
                    self.target_field, origin.target_start, origin.target_end,
                    asof=asof,
                )[self.target_column].reindex(origin.target_index)
                score = self._score(origin, prediction, actuals, asof)
                settled.append(SettlementRecord(origin, prediction, actuals, score))
            else:
                still_pending.append((origin, prediction))
        self._pending = still_pending
        return settled

    def _score(self, origin: Origin, prediction: pd.DataFrame,
               actuals: pd.Series, asof) -> float:
        """Score one settled origin. Subclasses may use market state at
        ``asof`` (always availability-checked through the portal)."""
        return self.objective.calculate(actuals, prediction)
