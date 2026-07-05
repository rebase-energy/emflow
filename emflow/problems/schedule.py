"""IssueSchedule: when forecasts are issued and what they cover — as data.

An energy-forecasting problem's cadence (zipline's trading calendar, renamed):
each :class:`Origin` is one forecast issue — a clock time ``asof`` at which the
model acts, plus the ``target_index`` of timestamps it must forecast. Both the
event-driven and vectorized execution modes consume the same origins, so the
schedule is the single source of truth about timing.
"""

from __future__ import annotations

from dataclasses import dataclass
import typing as t

import pandas as pd


@dataclass(frozen=True)
class Origin:
    """One forecast issue: act at ``asof``, forecast ``target_index``.

    ``column`` scopes the origin to one column of the target field —
    multi-zone competitions (GEFCom2014 wind's 10 zones) score each zone
    separately, so each (task, zone) pair is its own origin."""

    asof: pd.Timestamp
    target_index: pd.DatetimeIndex
    column: t.Optional[str] = None

    @property
    def target_start(self) -> pd.Timestamp:
        return self.target_index[0]

    @property
    def target_end(self) -> pd.Timestamp:
        return self.target_index[-1]

    def __repr__(self):
        return (f"Origin(asof={self.asof}, targets={self.target_start} → "
                f"{self.target_end}, n={len(self.target_index)})")


class IssueSchedule:
    """Generates :class:`Origin` s over an evaluation period.

    Construct with one of the classmethods; call :meth:`origins` with the
    period to evaluate over.
    """

    def __init__(self, origin_freq, cover_start, cover_end, target_freq,
                 origin_time=None):
        self.origin_freq = origin_freq
        self.cover_start = pd.Timedelta(cover_start)
        self.cover_end = pd.Timedelta(cover_end)
        self.target_freq = target_freq
        self.origin_time = origin_time
        if self.cover_end < self.cover_start:
            raise ValueError("cover_end must be >= cover_start")
        if self.cover_start <= pd.Timedelta(0):
            raise ValueError(
                "cover_start must be > 0: targets at or before the origin's asof "
                "would already be observable"
            )

    # -- constructors ---------------------------------------------------------

    @classmethod
    def hourly(cls, horizon="1h", target_freq="1h") -> "IssueSchedule":
        """One origin per hour, forecasting ``asof + horizon`` (rolling 1-step)."""
        return cls(origin_freq="1h", cover_start=horizon, cover_end=horizon,
                   target_freq=target_freq)

    @classmethod
    def daily(cls, at="09:00", covers=("1D", "2D"), target_freq="1h") -> "IssueSchedule":
        """One origin per day at ``at`` (UTC), covering ``asof + covers[0]`` to
        ``asof + covers[1]`` at ``target_freq`` — the day-ahead pattern."""
        return cls(origin_freq="1D", cover_start=covers[0], cover_end=covers[1],
                   target_freq=target_freq, origin_time=at)

    @classmethod
    def single(cls, asof, target_index) -> "IssueSchedule":
        """A one-shot schedule with an explicit target index (GEFCom task style)."""
        return _ExplicitSchedule([Origin(pd.Timestamp(asof), pd.DatetimeIndex(target_index))])

    @classmethod
    def explicit(cls, origins: t.Sequence[Origin]) -> "IssueSchedule":
        return _ExplicitSchedule(list(origins))

    # -- origin generation -------------------------------------------------------

    def origins(self, start, end) -> t.List[Origin]:
        """All origins whose *targets* fall inside the half-open ``[start, end)``.

        ``start``/``end`` delimit the evaluation period (target time, not issue
        time). Half-open, so adjacent splits partition scored timestamps with
        no double-counted boundary point.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # Earliest origin whose first target could be >= start:
        first_asof = start - self.cover_start
        asofs = pd.date_range(first_asof.floor(self.origin_freq), end,
                              freq=self.origin_freq, tz=start.tz)
        if self.origin_time is not None:
            at = pd.Timedelta(self.origin_time + ":00" if len(self.origin_time) == 5
                              else self.origin_time)
            asofs = (asofs.normalize() + at).unique()

        out = []
        for asof in asofs:
            targets = pd.date_range(asof + self.cover_start, asof + self.cover_end,
                                    freq=self.target_freq)
            targets = targets[(targets >= start) & (targets < end)]
            if len(targets):
                out.append(Origin(asof, targets))
        return out


class _ExplicitSchedule(IssueSchedule):
    def __init__(self, origins):
        self._origins = sorted(origins, key=lambda o: o.asof)

    def origins(self, start=None, end=None):
        out = self._origins
        if start is not None:
            start = pd.Timestamp(start)
            out = [o for o in out if o.target_end >= start]
        if end is not None:
            end = pd.Timestamp(end)
            out = [o for o in out if o.target_start < end]
        return list(out)
