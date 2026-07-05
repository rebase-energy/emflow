"""rebase-grid API-backed fields: live European grid data as emflow Fields.

The `rebase-grid <https://grid.rebase.energy>`_ API serves per-bidding-zone
power-system time series (ENTSO-E and other TSO sources). This module turns
them into :class:`~emflow.data.field.Field` s, with a local parquet cache so
problems built on live data stay cheap, reproducible and runnable offline
(hillclimb candidate sandboxes read the cache and never need the API key).

Variables (each becomes a single-column ``actual`` field named after itself):

* ``demand`` — total system load (MW), from ``/v1/zone/{zone}/consumption``
* ``wind``   — wind generation (MW), from ``/v1/zone/{zone}/production``
* ``solar``  — solar generation (MW), from ``/v1/zone/{zone}/production``

Environment:

* ``EMFLOW_GRID_API_KEY`` (or ``GRID_API_KEY``) — API key; only needed on a
  cache miss.
* ``EMFLOW_GRID_CACHE`` — cache root (default ``~/.cache/emflow/grid``).
* ``EMFLOW_GRID_OFFLINE=1`` — forbid network; a cache miss becomes an error.

The store behind the API is bitemporal, but the endpoints currently expose
only the latest revision, so fields use a constant ``availability_lag``
instead of a knowledge column. When the API grows an ``asof`` parameter,
switch to ``knowledge_col`` for fully honest revision-aware backtests.
"""

from __future__ import annotations

import os
import time
import typing as t
from pathlib import Path

import pandas as pd

from ..field import Field

BASE_URL = os.environ.get("EMFLOW_GRID_BASE_URL", "https://grid.rebase.energy/api")

#: variable -> (endpoint, query params)
VARIABLES: t.Dict[str, t.Tuple[str, t.Dict[str, str]]] = {
    "demand": ("consumption", {}),
    "wind": ("production", {"fuel": "wind"}),
    "solar": ("production", {"fuel": "solar"}),
}


def cache_root() -> Path:
    return Path(os.environ.get("EMFLOW_GRID_CACHE",
                               Path.home() / ".cache" / "emflow" / "grid"))


def _offline() -> bool:
    return os.environ.get("EMFLOW_GRID_OFFLINE", "") not in ("", "0")


def _api_key() -> t.Optional[str]:
    return os.environ.get("EMFLOW_GRID_API_KEY") or os.environ.get("GRID_API_KEY")


def _get_json(url: str, params: dict, api_key: str, retries: int = 3) -> dict:
    import requests

    for attempt in range(retries):
        resp = requests.get(url, params=params,
                            headers={"X-API-Key": api_key}, timeout=120)
        if resp.status_code in (429, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt * (int(resp.headers.get("Retry-After", 0)) or 2))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def fetch_series(zone: str, variable: str, start, end, *,
                 resolution: str = "1H", api_key: t.Optional[str] = None) -> pd.DataFrame:
    """Fetch one (zone, variable) series over half-open ``[start, end)``.

    Returns a single-column DataFrame named ``variable`` on a tz-aware UTC
    DatetimeIndex. Fetches in ~1-year chunks; missing values stay NaN.
    """
    if variable not in VARIABLES:
        raise KeyError(f"unknown variable {variable!r}; known: {sorted(VARIABLES)}")
    key = api_key or _api_key()
    if key is None:
        raise RuntimeError(
            f"no API key for rebase-grid (needed to fetch {variable}/{zone}); "
            "set EMFLOW_GRID_API_KEY (or GRID_API_KEY)"
        )
    endpoint, extra = VARIABLES[variable]
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=366), end)
        payload = _get_json(
            f"{BASE_URL}/v1/zone/{zone}/{endpoint}",
            {"start": chunk_start.isoformat(), "end": chunk_end.isoformat(),
             "resolution": resolution, **extra},
            key,
        )
        data = payload["data"]
        df = pd.DataFrame(data).set_index("datetime")
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="datetime")
        frames.append(df)
        chunk_start = chunk_end

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # One value column per variable: consumption serves load_mw; production
    # with a fuel filter serves one column per fuel (sum handles zones that
    # split e.g. wind into subcategories).
    out = out.sum(axis=1, min_count=1).to_frame(variable)
    return out[(out.index >= start) & (out.index < end)]


def load_series(zone: str, variable: str, start, end, *,
                resolution: str = "1H") -> pd.DataFrame:
    """Cache-first :func:`fetch_series`: hit the API only on a cache miss.

    The cache file is keyed by (zone, variable, start, end, resolution), so a
    problem with a pinned ``data_end`` always reads identical data — and
    sandboxed evaluations need neither network nor the API key.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    path = (cache_root() /
            f"{zone}_{variable}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_{resolution}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    if _offline():
        raise RuntimeError(
            f"EMFLOW_GRID_OFFLINE is set and {path} is not cached — pre-warm "
            f"the cache by loading the problem once with network access"
        )
    df = fetch_series(zone, variable, start, end, resolution=resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return df


def grid_field(zone: str, variable: str, start, end, *,
               availability_lag: str = "2h", resolution: str = "1H") -> Field:
    """A (zone, variable) series as an ``actual`` Field.

    ``availability_lag`` models ENTSO-E publication delay: at clock time
    ``asof`` the feed serves values stamped up to ``asof - lag``.
    """
    frame = load_series(zone, variable, start, end, resolution=resolution)
    return Field(
        name=variable,
        frame=frame,
        kind="actual",
        availability_lag=availability_lag,
        description=f"{variable} for zone {zone} (MW, {resolution}), rebase-grid API",
    )
