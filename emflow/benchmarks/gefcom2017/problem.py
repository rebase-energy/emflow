"""GEFCom2017 — hierarchical probabilistic load forecasting (ISO New England).

The qualifying match of GEFCom2017 (Hong, Xie & Black, IJF 2019): forecast the
9 deciles of hourly demand for the 8 ISO-NE zones plus the MASS and TOTAL
aggregates, one-to-two months ahead, ex-ante (no future weather). Six rounds:

    round 1: due 2016-12-15 → January 2017      round 4: due 2017-01-31 → March
    round 2: due 2016-12-31 → February          round 5: due 2017-02-14 → March
    round 3: due 2017-01-15 → February          round 6: due 2017-02-28 → April

One origin per (round, zone) — 60 in total. Rounds 1-3 (Jan/Feb targets) are
the ``validation`` split; rounds 4-6 (Mar/Apr) the ``holdout``. Scored by
pinball loss. The official ranking used relative-improvement-vs-benchmark, so
no directly comparable published pinball table exists — ``reference_scores``
is empty; beat the benchmark analyzers instead.

Data: local build cache (``scripts/build_gefcom2017.py``) or
``rb://dataset/rebase-energy/gefcom2017`` (+ ``-private``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import emflow as ef
from emflow.data import DataFeed
from emflow.envs import ForecastEnv
from emflow.problems.schedule import Origin

QUANTILES = tuple(q / 10 for q in range(1, 10))

PUBLIC_REPO = "rb://dataset/rebase-energy/gefcom2017"
PRIVATE_REPO = "rb://dataset/rebase-energy/gefcom2017-private"
LOCAL_BUILD = Path.home() / ".cache" / "emflow" / "gefcom2017" / "build"

ZONES = ["CT", "ME", "NEMASSBOST", "NH", "RI", "SEMASS", "VT", "WCMASS",
         "MASS", "TOTAL"]

#: (due date 23:59 EST ≈ asof next midnight UTC-5; kept tz-naive local, next day) → target month
ROUNDS = [
    ("2016-12-16", "2017-01"),
    ("2017-01-01", "2017-02"),
    ("2017-01-16", "2017-02"),
    ("2017-02-01", "2017-03"),
    ("2017-02-15", "2017-03"),
    ("2017-03-01", "2017-04"),
]

SPLITS = ef.Splits(
    train_end="2017-01-01",
    validation=("2017-01-01", "2017-03-01"),
    holdout=("2017-03-01", "2017-05-01"),
)


def round_origins():
    origins = []
    for asof, month in ROUNDS:
        start = pd.Timestamp(month)
        idx = pd.date_range(start, start + pd.DateOffset(months=1),
                            freq="1h", inclusive="left")
        for zone in ZONES:
            origins.append(Origin(pd.Timestamp(asof), idx, column=zone))
    return origins


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


def _load_split_dataset(kind: str) -> ef.Dataset:
    if (LOCAL_BUILD / kind / "rebase.yaml").exists():
        return _dataset_from_dir(LOCAL_BUILD / kind)
    return ef.Dataset.from_manifest(PUBLIC_REPO if kind == "public" else PRIVATE_REPO)


def load_dataset() -> ef.Dataset:
    dataset = _load_split_dataset("public")
    try:
        private = _load_split_dataset("private")
    except Exception:
        return dataset
    for name, priv in private.fields.items():
        pub = dataset.field(name)
        merged = pd.concat([pub.frame, priv.frame]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        dataset.add(ef.Field(name=name, frame=merged, kind=pub.kind,
                             availability_lag=pub.availability_lag,
                             description=pub.description))
    return dataset


def make_env(problem: ef.Problem, split: str) -> ForecastEnv:
    return ForecastEnv(
        feed=DataFeed(problem.load_dataset()),
        origins=problem.origins(split),
        target_field="load",
        objective=problem.objective,
        train_end=problem.splits.train_end,
        quantiles=QUANTILES,
    )


def get_problem() -> ef.Problem:
    return ef.Problem(
        name="gefcom2017",
        dataset=load_dataset,
        make_env=make_env,
        objective=ef.Objective(ef.PinballLoss(QUANTILES)),
        schedule=ef.IssueSchedule.explicit(round_origins()),
        splits=SPLITS,
        description=__doc__,
    )
