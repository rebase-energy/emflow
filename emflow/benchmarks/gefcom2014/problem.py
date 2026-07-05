"""GEFCom2014 — probabilistic load, price, wind and solar forecasting.

The Global Energy Forecasting Competition 2014 (Hong et al., IJF 2016): 15
incremental tasks per track, 99 quantiles per hour, scored by pinball loss.
Tasks 1–3 were trial runs (our ``validation`` split); tasks 4–15 were the
officially scored evaluation weeks (our ``holdout``).

Variants: ``gefcom2014:load`` (hourly utility load; 25 temperature stations,
deliberately NO future temperature), ``gefcom2014:price`` (zonal price given
load forecasts), ``gefcom2014:wind`` (10 farms, ECMWF U/V — one origin per
task × zone), ``gefcom2014:solar`` (3 plants, 12 ECMWF variables).

The competition's incremental data releases are reproduced with bitemporal
availability: each target row carries the knowledge time of the task release
that revealed it, and predictor rows are forecast fields issued at their
task's origin — the DataFeed serves exactly the information set participants
had at each task.

Data: local build cache (``scripts/build_gefcom2014.py``) or
``rb://dataset/rebase-energy/gefcom2014`` (+ ``-private`` holdout releases).
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import pandas as pd

import emflow as ef
from emflow.data import DataFeed
from emflow.envs import ForecastEnv
from emflow.problems.schedule import Origin

QUANTILES = tuple(q / 100 for q in range(1, 100))

PUBLIC_REPO = "rb://dataset/rebase-energy/gefcom2014"
PRIVATE_REPO = "rb://dataset/rebase-energy/gefcom2014-private"
LOCAL_BUILD = Path.home() / ".cache" / "emflow" / "gefcom2014" / "build"

HOLDOUT_FIRST_TASK = 4

TRACKS = {
    "load": {"target_field": "load_targets", "zones": ["load"]},
    "price": {"target_field": "price_targets", "zones": ["price"]},
    "wind": {"target_field": "wind_targets", "zones": [f"z{i}" for i in range(1, 11)]},
    "solar": {"target_field": "solar_targets", "zones": [f"z{i}" for i in range(1, 4)]},
}


def _tasks(track: str) -> pd.DataFrame:
    with importlib.resources.path(__package__, "tasks.csv") as path:
        tasks = pd.read_csv(path, parse_dates=["asof", "target_start", "target_end"])
    return tasks[tasks["track"] == track].sort_values("task")


def task_origins(track: str):
    """One origin per (task, zone): GEFCom2014 scored each zone separately."""
    zones = TRACKS[track]["zones"]
    origins = []
    for _, row in _tasks(track).iterrows():
        idx = pd.date_range(row["target_start"], row["target_end"], freq="1h")
        for zone in zones:
            # row["asof"], not row.asof — the latter is pandas' Series.asof method
            origins.append(Origin(row["asof"], idx, column=zone))
    return origins


def splits(track: str) -> ef.Splits:
    tasks = _tasks(track)
    t1 = tasks.iloc[0]
    t_hold = tasks[tasks.task == HOLDOUT_FIRST_TASK].iloc[0]
    t_last = tasks.iloc[-1]
    return ef.Splits(
        train_end=t1.target_start,
        validation=(t1.target_start, t_hold.target_start),
        holdout=(t_hold.target_start, t_last.target_end + pd.Timedelta("1h")),
    )


# -- data ------------------------------------------------------------------------------


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
                                knowledge_col=spec.get("knowledge_col"),
                                description=spec.get("description"))
    return ef.Dataset(name=manifest["name"], description=manifest.get("description"),
                      fields=fields)


def load_dataset() -> ef.Dataset:
    """Public releases, with holdout releases merged in when (and only when)
    the private repo/build is readable — the verifier, not agents."""
    dataset = _load_split_dataset("public")
    try:
        private = _load_split_dataset("private")
    except Exception:
        return dataset
    for name, priv in private.fields.items():
        pub = dataset.field(name)
        merged = pd.concat([pub.frame, priv.frame]).sort_index()
        if priv.kind == "actual":
            merged = merged.sort_values("knowledge_time", kind="stable").sort_index(kind="stable")
        dataset.add(ef.Field(name=name, frame=merged, kind=pub.kind,
                             availability_lag=pub.availability_lag,
                             knowledge_col=pub.knowledge_col,
                             description=pub.description))
    return dataset


def reference_scores(track: str):
    """Mean pinball over the scored evaluation weeks, from the official
    provisional leaderboard (teams scored in every week)."""
    with importlib.resources.path(__package__, "leaderboard.csv") as path:
        lb = pd.read_csv(path)
    lb = lb[lb["track"] == track].sort_values("rank")
    return [ef.RefScore(int(r["rank"]), r["team"], float(r["pinball"]))
            for _, r in lb.iterrows()]


# -- factories ---------------------------------------------------------------------------


def _make_env(track: str):
    def make_env(problem: ef.Problem, split: str) -> ForecastEnv:
        return ForecastEnv(
            feed=DataFeed(problem.load_dataset()),
            origins=problem.origins(split),
            target_field=TRACKS[track]["target_field"],
            objective=problem.objective,
            train_end=problem.splits.train_end,
            quantiles=QUANTILES,
        )
    return make_env


def _get_problem(track: str) -> ef.Problem:
    return ef.Problem(
        name=f"gefcom2014:{track}",
        dataset=load_dataset,
        make_env=_make_env(track),
        objective=ef.Objective(ef.PinballLoss(QUANTILES)),
        schedule=ef.IssueSchedule.explicit(task_origins(track)),
        splits=splits(track),
        description=__doc__,
        reference_scores=reference_scores(track),
    )


def list_problem_variants():
    return ["load", "price", "wind", "solar"]


def get_problem_load() -> ef.Problem:
    return _get_problem("load")


def get_problem_price() -> ef.Problem:
    return _get_problem("price")


def get_problem_wind() -> ef.Problem:
    return _get_problem("wind")


def get_problem_solar() -> ef.Problem:
    return _get_problem("solar")
