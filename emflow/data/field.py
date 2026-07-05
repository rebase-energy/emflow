"""Fields: time series with explicit availability semantics.

A :class:`Field` is one logical series of a dataset plus the rule for *when each
value becomes knowable*. This is the primitive that makes look-ahead leakage
impossible by construction: the :class:`~emflow.data.portal.DataPortal` only ever
serves values whose availability time is at or before the simulation clock.

Three kinds of field:

``actual``
    Observations stamped by measurement time. A value stamped ``t`` becomes
    knowable at ``t + availability_lag`` (meter readings arrive late; a lag of
    ``0`` means "knowable the moment it happens").

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
    description:
        Optional human-readable description (surfaced in manifests).
    """

    name: str
    frame: pd.DataFrame
    kind: str = "actual"
    availability_lag: t.Union[str, pd.Timedelta] = "0h"
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
            if not self.frame.index.is_monotonic_increasing:
                self.frame = self.frame.sort_index()
        elif self.kind == "forecast":
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
