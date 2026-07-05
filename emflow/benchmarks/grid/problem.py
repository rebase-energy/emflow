"""European grid autoregressive forecasting on live rebase-grid data.

One problem per (variable, bidding zone): forecast the next 24 hours of
``demand`` (total system load), ``wind`` or ``solar`` (generation, MW) from
the series' own past — no weather data is provided, so skill has to come from
autoregressive structure (diurnal/weekly cycles, recent level).

Origins are hourly: at each ``asof`` the model forecasts ``asof+1h`` through
``asof+24h``. Actuals become knowable ``2h`` after their timestamp (ENTSO-E
publication delay), so at ``asof`` the newest observable value is from
``asof-2h``.

Data comes from the rebase-grid API (``grid.rebase.energy``) pinned at
``DATA_END`` and cached locally as parquet — evaluation is reproducible and
runs offline once the cache is warm. Override the pin with
``EMFLOW_GRID_DATA_END`` (ISO timestamp) to refresh problems to newer data.

Variants: ``grid:<variable>-<zone>`` for variables ``demand``/``wind``/
``solar`` and zones ``de-lu`` (DE-LU), ``se-se4`` (SE-SE4), ``dk-dk1``
(DK-DK1) — e.g. ``grid:demand-de-lu``, ``grid:wind-se-se4``,
``grid:solar-dk-dk1``. Zones are pilots chosen for full backfill depth on
all three variables; extend ``ZONES`` as rebase-grid backfills more zones.
"""

from __future__ import annotations

import os

import pandas as pd

import emflow as ef
from emflow.data import DataFeed
from emflow.data.io.grid import grid_field
from emflow.envs import ForecastEnv

DATA_END = pd.Timestamp(os.environ.get("EMFLOW_GRID_DATA_END", "2026-07-01T00:00:00Z"))
HISTORY = pd.Timedelta(days=730)
AVAILABILITY_LAG = "2h"
HOLDOUT_DAYS = 21
VALIDATION_DAYS = 42

ZONES = {"de-lu": "DE-LU", "se-se4": "SE-SE4", "dk-dk1": "DK-DK1"}
TARGET_VARIABLES = ("demand", "wind", "solar")

SPLITS = ef.Splits(
    train_end=DATA_END - pd.Timedelta(days=HOLDOUT_DAYS + VALIDATION_DAYS),
    validation=(DATA_END - pd.Timedelta(days=HOLDOUT_DAYS + VALIDATION_DAYS),
                DATA_END - pd.Timedelta(days=HOLDOUT_DAYS)),
    holdout=(DATA_END - pd.Timedelta(days=HOLDOUT_DAYS), DATA_END),
)


def parse_variant(variant: str) -> tuple[str, str]:
    """``"demand-de-lu"`` -> ``("demand", "DE-LU")``."""
    variable, _, zone_slug = variant.partition("-")
    if variable not in TARGET_VARIABLES or zone_slug not in ZONES:
        raise KeyError(
            f"unknown grid variant {variant!r}; "
            f"variables: {TARGET_VARIABLES}, zones: {sorted(ZONES)}"
        )
    return variable, ZONES[zone_slug]


def _description(variable: str, zone: str) -> str:
    what = {
        "demand": "total system load",
        "wind": "wind power generation",
        "solar": "solar power generation",
    }[variable]
    return (
        f"Autoregressive forecasting of {what} (MW, hourly) for European "
        f"bidding zone {zone}, on live data from the rebase-grid API "
        f"(ENTSO-E sourced), pinned at {DATA_END.isoformat()}.\n\n"
        f"Target field: {variable!r} (single column, hourly, ~2 years of "
        f"history). Actuals become knowable {AVAILABILITY_LAG} after their "
        f"timestamp. At each hourly origin, forecast the next 24 hours "
        f"(asof+1h .. asof+24h). Scored by MAE (MW). No weather/NWP data is "
        f"available — use the target's own past (lags, rolling statistics, "
        f"calendar structure).\n\n"
        f"Splits: train < {SPLITS.train_end.isoformat()}, validation "
        f"{SPLITS.validation[0].date()} .. {SPLITS.validation[1].date()}, "
        f"holdout {SPLITS.holdout[0].date()} .. {SPLITS.holdout[1].date()}."
    )


def _load_dataset(variant: str) -> ef.Dataset:
    from emflow.problems.registry import ProblemNotIngestedError

    variable, zone = parse_variant(variant)
    field = grid_field(zone, variable, DATA_END - HISTORY, DATA_END,
                       availability_lag=AVAILABILITY_LAG)
    first_valid = field.frame[variable].first_valid_index()
    if first_valid is None or first_valid > SPLITS.train_end - pd.Timedelta(days=180):
        raise ProblemNotIngestedError(
            f"grid:{variant}: rebase-grid has {variable} for {zone} only from "
            f"{first_valid} — not enough history before train_end "
            f"{SPLITS.train_end} (need >= 180 days)"
        )
    return ef.Dataset(
        name=f"grid-{variant}",
        description=f"{variable} for {zone} from the rebase-grid API",
        fields={variable: field},
    )


def make_env(problem: ef.Problem, split: str) -> ForecastEnv:
    variable = problem.name.split(":", 1)[1].partition("-")[0]
    return ForecastEnv(
        feed=DataFeed(problem.load_dataset()),
        origins=problem.origins(split),
        target_field=variable,
        objective=problem.objective,
        train_end=problem.splits.train_end,
    )


def _build_problem(variant: str) -> ef.Problem:
    variable, zone = parse_variant(variant)
    return ef.Problem(
        name=f"grid:{variant}",
        dataset=lambda: _load_dataset(variant),
        make_env=make_env,
        objective=ef.Objective(ef.MeanAbsoluteError()),
        schedule=ef.IssueSchedule(origin_freq="1h", cover_start="1h",
                                  cover_end="24h", target_freq="1h"),
        splits=SPLITS,
        description=_description(variable, zone),
    )


def list_problem_variants():
    return [f"{v}-{z}" for v in TARGET_VARIABLES for z in ZONES]


def _make_factory(variant: str):
    def factory() -> ef.Problem:
        return _build_problem(variant)
    factory.__name__ = f"get_problem_{variant}"
    return factory


for _variant in list_problem_variants():
    globals()[f"get_problem_{_variant}"] = _make_factory(_variant)
del _variant
