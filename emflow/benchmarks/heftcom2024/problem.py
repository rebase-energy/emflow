"""HEFTCom2024 — the Hybrid Energy Forecasting and Trading Competition.

Forecast (and trade) the combined output of the Hornsea-1 offshore wind farm
and East-England (PES-10) solar in Great Britain. Participants submitted, every
day at 09:20 UTC, nine quantiles (10%..90%) of half-hourly generation for the
next day-ahead market delivery day, plus a market bid. The official evaluation
ran 2024-02-20 → 2024-05-19; 26 teams reached the final leaderboard.

Variants
--------
``heftcom2024:forecasting``
    Quantile forecasting scored by pinball loss — the Forecasting Track.
``heftcom2024:trading``
    Day-ahead bidding scored by total revenue under single-imbalance-price
    settlement — the Trading Track.

Data resolves from the local build cache
(``~/.cache/emflow/heftcom2024/build``, produced by
``scripts/build_heftcom2024.py``) when present, else from the HF Hub repos
``rb://dataset/rebase-energy/heftcom2024`` (public) and ``...-private``
(holdout labels, verifier-only). Without access to the private repo the
``holdout`` split cannot settle — that is the anti-cheat design, not a bug.

References: Browell et al., arXiv:2507.01579; data DOI 10.5281/zenodo.13950764.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import pandas as pd

import emflow as ef
from emflow.data import DataPortal
from emflow.envs import ForecastEnv, TradingEnv
from emflow.problems.metrics import TradingRevenue
from emflow.problems.schedule import Origin

QUANTILES = tuple(q / 10 for q in range(1, 10))
TARGET_FIELD = "energy"
TARGET_COLUMN = "total_generation_mwh"

PUBLIC_REPO = "rb://dataset/rebase-energy/heftcom2024"
PRIVATE_REPO = "rb://dataset/rebase-energy/heftcom2024-private"
LOCAL_BUILD = Path.home() / ".cache" / "emflow" / "heftcom2024" / "build"

#: Data span and official evaluation period (see scripts/build_heftcom2024.py).
DATA_START = pd.Timestamp("2020-09-20 00:00", tz="UTC")
COMPETITION_START = pd.Timestamp("2024-02-20 00:00", tz="UTC")
DATA_END = pd.Timestamp("2024-05-20 00:00", tz="UTC")

# Boundaries sit on market-day starts (23:00 UTC = midnight CET) so origins
# never straddle a split. The holdout is exactly the officially scored window:
# delivery days 2024-02-20 → 2024-05-19 (CET calendar).
SPLITS = ef.Splits(
    train_end="2024-01-31T23:00:00Z",
    validation=("2024-01-31T23:00:00Z", "2024-02-19T23:00:00Z"),
    holdout=("2024-02-19T23:00:00Z", "2024-05-19T22:00:00Z"),
)


# -- schedule -----------------------------------------------------------------------

MARKET_TZ = "Europe/Paris"  # GB day-ahead auction trades the CET calendar day


def market_day_origins(start=DATA_START, end=DATA_END):
    """One origin per day-ahead market delivery day.

    Submission at 09:20 UTC on day D covers delivery day D+1: the half-hourly
    settlement periods of the **CET (Europe/Paris) calendar day** — 23:00 →
    22:30 UTC in winter, 22:00 → 21:30 UTC in summer, 46 periods on the
    spring-forward day. Verified against the official trades archive (the
    first scored block is 2024-02-19 23:00 UTC; the 2024-03-31 delivery day
    has 46 periods).
    """
    days = pd.date_range(start.tz_convert(MARKET_TZ).normalize(),
                         end.tz_convert(MARKET_TZ).normalize(),
                         freq="1D", tz=MARKET_TZ)
    origins = []
    for day in days:
        # Bounds from the *local calendar* (naive date re-localized), so a DST
        # delivery day really has 46/50 periods instead of a fixed 48.
        naive = day.tz_localize(None).normalize()
        day_end = (naive + pd.Timedelta("1D")).tz_localize(MARKET_TZ)
        local = pd.date_range(day, day_end, freq="30min", inclusive="left")
        targets = pd.DatetimeIndex(local.tz_convert("UTC"))
        # 09:20 UTC on the calendar day before delivery day D — also computed
        # on the naive date so DST can't collapse two days onto one asof.
        asof = (naive - pd.Timedelta("1D")).tz_localize("UTC") + pd.Timedelta("9h20min")
        origins.append(Origin(asof, targets))
    return origins


# -- data ---------------------------------------------------------------------------

def _load_split_dataset(kind: str) -> ef.Dataset:
    if (LOCAL_BUILD / kind / "rebase.yaml").exists():
        return _dataset_from_dir(LOCAL_BUILD / kind)
    return ef.Dataset.from_manifest(PUBLIC_REPO if kind == "public" else PRIVATE_REPO)


def _dataset_from_dir(root: Path) -> ef.Dataset:
    import yaml

    manifest = yaml.safe_load((root / "rebase.yaml").read_text())
    fields = {}
    for name, spec in manifest["fields"].items():
        frame = pd.read_parquet(root / spec["path"])
        fields[name] = ef.Field(name=name, frame=frame, kind=spec.get("kind", "actual"),
                                availability_lag=spec.get("availability_lag", "0h"),
                                description=spec.get("description"))
    return ef.Dataset(name=manifest["name"], description=manifest.get("description"),
                      fields=fields)


def load_dataset() -> ef.Dataset:
    """Public data, with holdout labels merged in when (and only when) the
    private repo/build is readable — i.e. for the verifier, not for agents."""
    dataset = _load_split_dataset("public")
    try:
        private = _load_split_dataset("private")
    except Exception:
        return dataset
    for name in ("energy", "prices"):
        pub, priv = dataset.field(name), private.field(name)
        merged = pd.concat([pub.frame, priv.frame]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        dataset.add(ef.Field(name=name, frame=merged, kind=pub.kind,
                             availability_lag=pub.availability_lag,
                             description=pub.description))
    return dataset


# -- reference scores -----------------------------------------------------------------

def reference_scores(track: str = "forecasting"):
    """The official final leaderboard (26 teams), packaged with emflow."""
    with importlib.resources.path(__package__, "leaderboard.csv") as path:
        lb = pd.read_csv(path)
    if track == "forecasting":
        lb = lb.dropna(subset=["Pinball", "Forecasting Track Rank"])
        return [ef.RefScore(int(r["Forecasting Track Rank"]), r["Team"], float(r["Pinball"]))
                for _, r in lb.sort_values("Forecasting Track Rank").iterrows()]
    lb = lb.dropna(subset=["Revenue", "Trading Track Rank"])
    return [ef.RefScore(int(r["Trading Track Rank"]), r["Team"], float(r["Revenue"]))
            for _, r in lb.sort_values("Trading Track Rank").iterrows()]


# -- problem factories -----------------------------------------------------------------

def make_env(problem: ef.Problem, split: str) -> ForecastEnv:
    return ForecastEnv(
        portal=DataPortal(problem.load_dataset()),
        origins=problem.origins(split),
        target_field=TARGET_FIELD,
        objective=problem.objective,
        target_column=TARGET_COLUMN,
        train_end=problem.splits.train_end,
        quantiles=QUANTILES,
    )


def make_trading_env(problem: ef.Problem, split: str) -> TradingEnv:
    return TradingEnv(
        portal=DataPortal(problem.load_dataset()),
        origins=problem.origins(split),
        target_field=TARGET_FIELD,
        objective=problem.objective,
        target_column=TARGET_COLUMN,
        train_end=problem.splits.train_end,
        prices_field="prices",
        da_column="da_price",
        ss_column="ss_price",
    )


def list_problem_variants():
    return ["forecasting", "trading"]


def get_problem_forecasting() -> ef.Problem:
    return ef.Problem(
        name="heftcom2024:forecasting",
        dataset=load_dataset,
        make_env=make_env,
        objective=ef.Objective(ef.PinballLoss(QUANTILES)),
        schedule=ef.IssueSchedule.explicit(market_day_origins()),
        splits=SPLITS,
        description=__doc__,
        reference_scores=reference_scores("forecasting"),
    )


def get_problem_trading() -> ef.Problem:
    return ef.Problem(
        name="heftcom2024:trading",
        dataset=load_dataset,
        make_env=make_trading_env,
        objective=ef.Objective(TradingRevenue(), lower_is_better=False),
        schedule=ef.IssueSchedule.explicit(market_day_origins()),
        splits=SPLITS,
        description=__doc__,
        reference_scores=reference_scores("trading"),
    )
