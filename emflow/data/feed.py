"""DataFeed: the only object that hands out data during a run.

Everything a model sees while being evaluated flows through here, bound to the
simulation clock (*asof*). Availability rules live on the
:class:`~emflow.data.field.Field`; the feed enforces them:

* actuals — values stamped ``t`` are served iff ``t + availability_lag <= asof``;
* forecasts — runs issued at ``i`` are served iff ``i + availability_lag <= asof``,
  and per valid_time the **latest** such run wins;
* statics — always served.

With the feed as the sole gate, look-ahead leakage is impossible by
construction rather than by runner discipline.
"""

from __future__ import annotations

import typing as t

import pandas as pd

from .dataset import Dataset
from .field import Field, ISSUE_LEVEL, VALID_LEVEL


class DataFeed:
    """Point-in-time access to a :class:`Dataset`."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    # -- core queries -----------------------------------------------------------

    def history(self, asof, name: str, window=None, strict=False) -> pd.DataFrame:
        """Actual values of field ``name`` knowable at ``asof``.

        For bitemporal fields (``knowledge_col`` set) the filter runs on the
        knowledge axis and, per measurement timestamp, the **latest revision**
        already knowable wins — a backtest sees preliminary values exactly as
        a live participant did. The knowledge column is dropped from the
        returned frame.

        ``window`` (optional, anything ``pd.Timedelta`` accepts) limits the
        result to the trailing window before the availability cutoff — an
        efficiency knob, never a leakage one.

        ``strict`` excludes the cutoff instant itself (``<`` instead of
        ``<=``). Training views use this so a training boundary placed exactly
        on the first scored target never serves it.
        """
        f = self._actual(name)
        asof = pd.Timestamp(asof)
        tz = f.frame.index.tz
        if tz is not None and asof.tz is None:
            asof = asof.tz_localize(tz)

        if f.knowledge_col is not None:
            kt = f.frame[f.knowledge_col]
            knowable = (kt < asof) if strict else (kt <= asof)
            out = f.frame[knowable.to_numpy()]
            # Frame is sorted by (timestamp, knowledge_time): the last
            # occurrence per timestamp is the latest knowable revision.
            out = out[~out.index.duplicated(keep="last")]
            out = out.drop(columns=[f.knowledge_col])
            if window is not None:
                out = out.loc[asof - pd.Timedelta(window):]
            return out

        cutoff = asof - f.availability_lag
        out = f.frame.loc[:cutoff]
        if strict and len(out) and out.index[-1] == cutoff:
            out = out.iloc[:-1]
        if window is not None:
            out = out.loc[cutoff - pd.Timedelta(window):]
        return out

    def forecasts(self, asof, name: str, columns=None) -> pd.DataFrame:
        """Latest forecast run of field ``name`` knowable at ``asof``.

        Returns a frame indexed by ``valid_time`` where, for each valid_time,
        the value comes from the most recent run with
        ``issue_time + availability_lag <= asof``. An ``issue_time`` column
        records which run each row came from.
        """
        f = self.dataset.field(name)
        if f.kind != "forecast":
            raise ValueError(f"field {name!r} is {f.kind!r}, not a forecast field")
        asof = pd.Timestamp(asof)
        cutoff = asof - f.availability_lag

        issues = f.frame.index.get_level_values(ISSUE_LEVEL)
        avail = f.frame[issues <= cutoff]
        if avail.empty:
            empty = f.frame.iloc[:0].reset_index(level=ISSUE_LEVEL)
            return empty if columns is None else empty[[ISSUE_LEVEL, *columns]]

        # Frame is sorted by (issue, valid); the last occurrence per valid_time
        # therefore comes from the most recent available run.
        flat = avail.reset_index(level=ISSUE_LEVEL)
        flat = flat[~flat.index.duplicated(keep="last")].sort_index()
        if columns is not None:
            flat = flat[[ISSUE_LEVEL, *columns]]
        return flat

    def static(self, name: str) -> pd.DataFrame:
        f = self.dataset.field(name)
        if f.kind != "static":
            raise ValueError(f"field {name!r} is {f.kind!r}, not a static field")
        return f.frame

    def actuals_between(self, name: str, start, end, asof) -> pd.DataFrame:
        """Actuals of ``name`` in ``[start, end]`` — only if knowable at ``asof``.

        Used by environments to settle scores: returns the slice restricted to
        what the availability rule allows at ``asof`` (so an env can only score
        an origin once its targets have actually arrived).
        """
        hist = self.history(asof, name)
        tz = hist.index.tz
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if tz is not None:
            start = start.tz_localize(tz) if start.tz is None else start
            end = end.tz_localize(tz) if end.tz is None else end
        return hist.loc[start:end]

    # -- views ------------------------------------------------------------------

    def view(self, asof, strict=False) -> "TimeView":
        """A feed bound to a fixed clock time (what observations wrap).

        ``strict=True`` makes the boundary exclusive — used for training views.
        """
        return TimeView(self, asof, strict=strict)

    # -- helpers ------------------------------------------------------------------

    def _actual(self, name: str) -> Field:
        f = self.dataset.field(name)
        if f.kind != "actual":
            raise ValueError(f"field {name!r} is {f.kind!r}, not an actual field")
        return f


class TimeView:
    """A :class:`DataFeed` frozen at one clock time.

    This is the whole world a model gets to see: observations and training
    views are ``TimeView`` s. There is deliberately no way to reach the
    underlying dataset or move the clock from here.
    """

    def __init__(self, feed: DataFeed, asof, strict=False):
        self._feed = feed
        self.asof = pd.Timestamp(asof)
        self.strict = strict

    def history(self, name: str, window=None) -> pd.DataFrame:
        return self._feed.history(self.asof, name, window=window, strict=self.strict)

    def forecasts(self, name: str, columns=None) -> pd.DataFrame:
        return self._feed.forecasts(self.asof, name, columns=columns)

    def static(self, name: str) -> pd.DataFrame:
        return self._feed.static(name)

    @property
    def fields(self):
        return sorted(self._feed.dataset.fields)

    def __repr__(self):
        return f"TimeView(asof={self.asof}, fields={self.fields})"
