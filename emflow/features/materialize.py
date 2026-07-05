"""Point-in-time-correct feature materialization.

Two paths that must (and are tested to) agree:

* :func:`materialize` — batch: one matrix for many origins at once, indexed by
  ``(asof, target_time)``. This is what makes the vectorized execution mode
  fast: all origins in one shot, masked by availability.
* :func:`materialize_observation` — one origin, built *purely from the
  observation's view API* (which already enforces availability). Used by
  :class:`~emflow.models.predictor.FeaturePredictor` in event mode.

Both produce NaN wherever a feature would need data outside the origin's
information set — never a peeked value.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

from ..data import DataPortal, TimeView
from .spec import Calendar, FeatureSpec, ForecastField, Lag, Rolling

INDEX_NAMES = ("asof", "target_time")


def _column(frame: pd.DataFrame, column: t.Optional[str], what: str) -> pd.Series:
    if column is not None:
        return frame[column]
    if frame.shape[1] != 1:
        raise ValueError(f"{what}: field has columns {list(frame.columns)}; specify column=")
    return frame.iloc[:, 0]


def _calendar_values(part: str, times: pd.DatetimeIndex) -> np.ndarray:
    return getattr(times, part).to_numpy(dtype=float)


def _align_forecast(fc: pd.DataFrame, col: str, targets: pd.DatetimeIndex,
                    interp: t.Optional[str]) -> np.ndarray:
    """Align one forecast column onto target times, optionally interpolating
    between the run's native steps (uses only the knowable run — no leak)."""
    series = fc[col]
    if interp is None:
        return series.reindex(targets).to_numpy(dtype=float)
    union = series.index.union(targets)
    return (series.reindex(union).interpolate(method=interp, limit_area="inside")
            .reindex(targets).to_numpy(dtype=float))


# -- batch path -------------------------------------------------------------------


def materialize(portal: DataPortal, features: t.Sequence[FeatureSpec], origins) -> pd.DataFrame:
    """Feature matrix for all ``origins``, indexed by ``(asof, target_time)``."""
    asofs, targets, origin_slices = [], [], []
    pos = 0
    for o in origins:
        n = len(o.target_index)
        asofs.append(pd.DatetimeIndex([o.asof] * n))
        targets.append(o.target_index)
        origin_slices.append((o, slice(pos, pos + n)))
        pos += n
    asof_arr = asofs[0].append(asofs[1:]) if len(asofs) > 1 else asofs[0]
    target_arr = targets[0].append(targets[1:]) if len(targets) > 1 else targets[0]
    index = pd.MultiIndex.from_arrays([asof_arr, target_arr], names=INDEX_NAMES)

    cols: t.Dict[str, np.ndarray] = {}
    for spec in features:
        if isinstance(spec, Lag):
            field = portal.dataset.field(spec.field)
            series = _column(field.frame, spec.column, f"Lag({spec.field!r})")
            for name, lag in zip(spec.names(), spec.lags):
                lag_td = pd.Timedelta(lag)
                lookup = target_arr - lag_td
                vals = series.reindex(lookup).to_numpy(dtype=float, copy=True)
                knowable = (lookup + field.availability_lag) <= asof_arr
                vals[~np.asarray(knowable)] = np.nan
                cols[name] = vals

        elif isinstance(spec, Rolling):
            field = portal.dataset.field(spec.field)
            series = _column(field.frame, spec.column, f"Rolling({spec.field!r})")
            rolled = series.rolling(spec.window).agg(spec.agg).dropna()
            cutoffs = asof_arr - field.availability_lag
            if rolled.empty:
                vals = np.full(len(index), np.nan)
            else:
                vals = rolled.reindex(cutoffs, method="ffill").to_numpy(dtype=float)
            cols[spec.names()[0]] = vals

        elif isinstance(spec, ForecastField):
            blocks = {name: np.full(len(index), np.nan) for name in spec.names()}
            for origin, sl in origin_slices:
                fc = portal.forecasts(origin.asof, spec.field,
                                      columns=list(spec.columns) or None)
                if fc.empty:
                    continue
                for name, col in zip(spec.names(), spec.columns):
                    blocks[name][sl] = _align_forecast(fc, col, origin.target_index,
                                                       spec.interp)
            cols.update(blocks)

        elif isinstance(spec, Calendar):
            for name, part in zip(spec.names(), spec.parts):
                cols[name] = _calendar_values(part, target_arr)

        else:
            raise TypeError(f"unknown feature spec {type(spec).__name__}")

    return pd.DataFrame(cols, index=index)


# -- single-origin path (built on the view API only) ---------------------------------


def materialize_observation(view: TimeView, features: t.Sequence[FeatureSpec],
                            target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Feature matrix for one origin, using only what the view serves.

    The view's availability enforcement *is* the masking: a lag reaching past
    the information set simply isn't in ``view.history()`` and reindexes to
    NaN. Index matches :func:`materialize`'s rows for the same origin.
    """
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([view.asof] * len(target_index)), target_index],
        names=INDEX_NAMES,
    )

    cols: t.Dict[str, np.ndarray] = {}
    for spec in features:
        if isinstance(spec, Lag):
            hist = view.history(spec.field)
            series = _column(hist, spec.column, f"Lag({spec.field!r})")
            for name, lag in zip(spec.names(), spec.lags):
                lookup = target_index - pd.Timedelta(lag)
                cols[name] = series.reindex(lookup).to_numpy(dtype=float)

        elif isinstance(spec, Rolling):
            hist = view.history(spec.field, window=None)
            series = _column(hist, spec.column, f"Rolling({spec.field!r})")
            rolled = series.rolling(spec.window).agg(spec.agg).dropna()
            val = rolled.iloc[-1] if len(rolled) else np.nan
            cols[spec.names()[0]] = np.full(len(index), float(val))

        elif isinstance(spec, ForecastField):
            fc = view.forecasts(spec.field, columns=list(spec.columns) or None)
            for name, col in zip(spec.names(), spec.columns):
                cols[name] = (np.full(len(index), np.nan) if fc.empty
                              else _align_forecast(fc, col, target_index, spec.interp))

        elif isinstance(spec, Calendar):
            for name, part in zip(spec.names(), spec.parts):
                cols[name] = _calendar_values(part, target_index)

        else:
            raise TypeError(f"unknown feature spec {type(spec).__name__}")

    return pd.DataFrame(cols, index=index)


def supervised_frame(portal: DataPortal, features: t.Sequence[FeatureSpec], origins,
                     target_field: str, target_column=None):
    """``(X, y)`` for supervised training over *training-period* origins.

    ``y`` reads actuals at target times directly — only ever call this on
    origins whose targets lie in the training split. Evaluation goes through
    the environment, which cannot leak.
    """
    X = materialize(portal, features, origins)
    field = portal.dataset.field(target_field)
    series = _column(field.frame, target_column, f"target {target_field!r}")
    y = pd.Series(series.reindex(X.index.get_level_values("target_time")).to_numpy(),
                  index=X.index, name=series.name)
    return X, y
