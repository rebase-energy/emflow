"""Fields: time series with explicit availability semantics.

A :class:`Field` is one logical series of a dataset plus the rule for *when each
value becomes knowable*. This is the primitive that makes look-ahead leakage
impossible by construction: the :class:`~emflow.data.portal.DataPortal` only ever
serves values whose availability time is at or before the simulation clock.

Three kinds of field:

``actual``
    Observations stamped by measurement time. Two availability rules:

    * constant lag (default): a value stamped ``t`` becomes knowable at
      ``t + availability_lag`` — the right model for settled-only archives;
    * **bitemporal** (``knowledge_col=...``): each row carries its own
      knowledge timestamp, and the same measurement time may appear multiple
      times with successive revisions (preliminary → settled). The portal
      serves, at any clock time, the latest revision already knowable — so a
      backtest sees the preliminary value exactly as a live participant did.

``forecast``
    Predictions stamped by *(issue_time, valid_time)* — e.g. NWP runs. The run
    issued at ``issue_time`` becomes knowable at ``issue_time +
    availability_lag`` (dissemination delay). At any clock time the portal
    serves, per valid_time, the latest run already issued. Using tomorrow's
    weather *actuals* as if they were forecasts is the classic energy-backtest
    leak; this kind exists to make that mistake unrepresentable.

``static``
    Time-invariant metadata (capacities, coordinates). Always available.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import typing as t

import numpy as np
import pandas as pd

KINDS = ("actual", "forecast", "static")

ISSUE_LEVEL = "issue_time"
VALID_LEVEL = "valid_time"


@dataclass
class Field:
    """One logical time series + its availability rule.

    Parameters
    ----------
    name:
        Field name, unique within a :class:`~emflow.data.dataset.Dataset`.
    frame:
        The data. ``actual``: a DataFrame with a tz-aware ``DatetimeIndex``
        (measurement time). ``forecast``: a DataFrame with a two-level
        MultiIndex ``(issue_time, valid_time)`` — use :meth:`Field.forecast`
        to build one from columns. ``static``: any DataFrame.
    kind:
        ``"actual"`` (default), ``"forecast"`` or ``"static"``.
    availability_lag:
        Delay between a value's timestamp (actuals) or a run's issue time
        (forecasts) and the moment it becomes knowable. Anything
        ``pd.Timedelta`` accepts; default ``"0h"``.
    knowledge_col:
        Actuals only: name of a column holding each row's knowledge timestamp
        (bitemporal storage — rows may repeat an index timestamp with
        revisions). Mutually exclusive with a non-zero ``availability_lag``.
    description:
        Optional human-readable description (surfaced in manifests).
    """

    name: str
    frame: pd.DataFrame
    kind: str = "actual"
    availability_lag: t.Union[str, pd.Timedelta] = "0h"
    knowledge_col: t.Optional[str] = None
    description: t.Optional[str] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Field {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if isinstance(self.frame, pd.Series):
            self.frame = self.frame.to_frame()
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError(f"Field {self.name!r}: frame must be a pandas DataFrame")
        self.availability_lag = pd.Timedelta(self.availability_lag)
        if self.availability_lag < pd.Timedelta(0):
            raise ValueError(f"Field {self.name!r}: availability_lag must be >= 0")

        if self.kind == "actual":
            if not isinstance(self.frame.index, pd.DatetimeIndex):
                raise TypeError(f"actual field {self.name!r} needs a DatetimeIndex")
            if self.knowledge_col is not None:
                if self.availability_lag != pd.Timedelta(0):
                    raise ValueError(
                        f"Field {self.name!r}: knowledge_col and a non-zero "
                        f"availability_lag are mutually exclusive — the "
                        f"knowledge column IS the availability rule"
                    )
                if self.knowledge_col not in self.frame.columns:
                    raise ValueError(
                        f"Field {self.name!r}: knowledge_col "
                        f"{self.knowledge_col!r} not in columns "
                        f"{list(self.frame.columns)}"
                    )
                kt = pd.to_datetime(self.frame[self.knowledge_col])
                if (kt.dt.tz is None) != (self.frame.index.tz is None):
                    raise ValueError(
                        f"Field {self.name!r}: knowledge_col and index must "
                        f"both be tz-aware or both tz-naive"
                    )
                if (kt < self.frame.index.to_series(index=kt.index)).any():
                    raise ValueError(
                        f"Field {self.name!r}: knowledge_time before the "
                        f"measurement timestamp — a value cannot be known "
                        f"before it happens"
                    )
                self.frame = self.frame.assign(**{self.knowledge_col: kt})
                order = np.lexsort([kt.to_numpy(), self.frame.index.to_numpy()])
                self.frame = self.frame.iloc[order]
            elif not self.frame.index.is_monotonic_increasing:
                self.frame = self.frame.sort_index()
        elif self.knowledge_col is not None:
            raise ValueError(f"Field {self.name!r}: knowledge_col only applies to actual fields")

        if self.kind == "forecast":
            idx = self.frame.index
            if not (isinstance(idx, pd.MultiIndex) and idx.nlevels == 2):
                raise TypeError(
                    f"forecast field {self.name!r} needs a (issue_time, valid_time) "
                    f"MultiIndex — build it with Field.forecast(...)"
                )
            if list(idx.names) != [ISSUE_LEVEL, VALID_LEVEL]:
                self.frame.index = idx.set_names([ISSUE_LEVEL, VALID_LEVEL])
            if not self.frame.index.is_monotonic_increasing:
                self.frame = self.frame.sort_index()

    # -- constructors ---------------------------------------------------------

    @classmethod
    def forecast(cls, name, frame, issue_col, valid_col, availability_lag="0h",
                 description=None) -> "Field":
        """Build a forecast field from a flat frame with issue/valid columns."""
        frame = frame.set_index([issue_col, valid_col])
        frame.index = frame.index.set_names([ISSUE_LEVEL, VALID_LEVEL])
        return cls(name=name, frame=frame, kind="forecast",
                   availability_lag=availability_lag, description=description)

    @classmethod
    def static(cls, name, frame, description=None) -> "Field":
        return cls(name=name, frame=frame, kind="static", description=description)

    # -- introspection --------------------------------------------------------

    @property
    def settlement_lag(self) -> pd.Timedelta:
        """How long after a timestamp its value is *finally* knowable.

        Environments use this to decide when an origin can settle. For
        constant-lag fields it's ``availability_lag``; for bitemporal fields
        it's the largest observed revision delay, so settlement waits for the
        settled (last) revision rather than scoring against a preliminary one.
        """
        if self.kind == "actual" and self.knowledge_col is not None:
            kt = self.frame[self.knowledge_col]
            if not len(kt):
                return pd.Timedelta(0)
            delays = kt - self.frame.index.to_series(index=kt.index)
            return delays.max()
        return self.availability_lag

    @property
    def start(self) -> t.Optional[pd.Timestamp]:
        if self.kind == "actual":
            return self.frame.index[0] if len(self.frame) else None
        if self.kind == "forecast":
            valid = self.frame.index.get_level_values(VALID_LEVEL)
            return valid.min() if len(valid) else None
        return None

    @property
    def end(self) -> t.Optional[pd.Timestamp]:
        if self.kind == "actual":
            return self.frame.index[-1] if len(self.frame) else None
        if self.kind == "forecast":
            valid = self.frame.index.get_level_values(VALID_LEVEL)
            return valid.max() if len(valid) else None
        return None

    def __repr__(self):
        span = f", {self.start} → {self.end}" if self.kind != "static" else ""
        return (f"Field({self.name!r}, kind={self.kind!r}, "
                f"lag={self.availability_lag}{span}, shape={self.frame.shape})")
