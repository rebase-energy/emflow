"""Declarative feature specs, materialized point-in-time-correctly.

The most common *accidental* leak is feature engineering done on the full
frame — lags and rolling means computed once over train+test. These specs are
instead materialized through the portal's availability rules (see
:mod:`emflow.features.materialize`): a lag that would reach past the origin's
information set comes out NaN, never as a peeked value.

Four spec types (deliberately minimal):

* :class:`Lag` — value of an actual field at ``target_time - lag``
* :class:`Rolling` — trailing-window aggregate of an actual field's *available*
  history at the origin
* :class:`ForecastField` — columns of the latest forecast run knowable at the
  origin, aligned to target times
* :class:`Calendar` — deterministic calendar encodings of the target time
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import typing as t

import pandas as pd

CALENDAR_PARTS = ("hour", "dayofweek", "month", "dayofyear")


@dataclass(frozen=True)
class Lag:
    """``field``'s value at ``target_time - lag``, if knowable at the origin."""

    field: str
    lags: t.Tuple[str, ...]
    column: t.Optional[str] = None

    def __post_init__(self):
        lags = (self.lags,) if isinstance(self.lags, str) else tuple(self.lags)
        object.__setattr__(self, "lags", lags)

    def names(self):
        return [f"{self.field}_lag_{lag}" for lag in self.lags]


@dataclass(frozen=True)
class Rolling:
    """Trailing-window ``agg`` of ``field`` over the history available at the
    origin (same value for every target time of one origin)."""

    field: str
    window: str
    agg: str = "mean"
    column: t.Optional[str] = None

    def names(self):
        return [f"{self.field}_roll_{self.window}_{self.agg}"]


@dataclass(frozen=True)
class ForecastField:
    """Columns of the latest run of forecast field ``field`` knowable at the
    origin, looked up at each target time.

    ``interp="time"`` time-interpolates the run onto target times that fall
    between its native steps (e.g. hourly NWP onto half-hourly settlement
    periods). Interpolation only ever uses the already-knowable run — it
    cannot leak."""

    field: str
    columns: t.Tuple[str, ...] = ()
    interp: t.Optional[str] = None

    def __post_init__(self):
        cols = (self.columns,) if isinstance(self.columns, str) else tuple(self.columns)
        object.__setattr__(self, "columns", cols)

    def names(self):
        return [f"{self.field}_{c}" for c in self.columns]


@dataclass(frozen=True)
class Calendar:
    """Calendar encodings of the target time (leak-free by construction)."""

    parts: t.Tuple[str, ...] = ("hour", "dayofweek")

    def __post_init__(self):
        parts = (self.parts,) if isinstance(self.parts, str) else tuple(self.parts)
        unknown = set(parts) - set(CALENDAR_PARTS)
        if unknown:
            raise ValueError(f"unknown calendar parts {sorted(unknown)}; "
                             f"available: {CALENDAR_PARTS}")
        object.__setattr__(self, "parts", parts)

    def names(self):
        return [f"cal_{p}" for p in self.parts]


FeatureSpec = t.Union[Lag, Rolling, ForecastField, Calendar]
