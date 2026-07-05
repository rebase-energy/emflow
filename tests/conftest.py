"""Shared toy problem: a deterministic hourly series with a day-ahead NWP field."""

import numpy as np
import pandas as pd
import pytest

from emflow.data import DataFeed, Dataset, Field
from emflow.envs import ForecastEnv
from emflow.problems import IssueSchedule, MeanAbsoluteError, Objective, Problem, Splits


TRAIN_END = pd.Timestamp("2025-03-01", tz="UTC")
VAL = (pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2025-03-08", tz="UTC"))
HOLDOUT = (pd.Timestamp("2025-03-08", tz="UTC"), pd.Timestamp("2025-03-15", tz="UTC"))


def toy_dataset(seed=0) -> Dataset:
    idx = pd.date_range("2025-01-01", "2025-03-15", freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    hour = idx.hour.to_numpy()
    y = 10.0 + 5.0 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.5, len(idx))
    target = pd.DataFrame({"power": y}, index=idx)

    rows = []
    for issue in pd.date_range("2025-01-01", "2025-03-15", freq="12h", tz="UTC"):
        for h in range(1, 49):
            valid = issue + pd.Timedelta(hours=h)
            rows.append((issue, valid, float(np.sin(2 * np.pi * valid.hour / 24))))
    nwp = pd.DataFrame(rows, columns=["issue", "valid", "signal"])

    return Dataset(
        name="toy",
        fields={
            "power": Field("power", target, availability_lag="0h"),
            "nwp": Field.forecast("nwp", nwp, issue_col="issue", valid_col="valid",
                                  availability_lag="6h"),
        },
    )


def make_toy_problem(schedule=None) -> Problem:
    dataset = toy_dataset()
    schedule = schedule or IssueSchedule.hourly(horizon="1h")
    splits = Splits(train_end=TRAIN_END, validation=VAL, holdout=HOLDOUT)

    def make_env(problem, split):
        return ForecastEnv(
            feed=DataFeed(problem.load_dataset()),
            origins=problem.origins(split),
            target_field="power",
            objective=problem.objective,
            train_end=problem.splits.train_end,
        )

    return Problem(
        name="toy:hourly",
        dataset=dataset,
        make_env=make_env,
        objective=Objective(MeanAbsoluteError()),
        schedule=schedule,
        splits=splits,
        description="synthetic hourly toy problem",
    )


@pytest.fixture
def toy_problem() -> Problem:
    return make_toy_problem()


@pytest.fixture
def toy_dayahead_problem() -> Problem:
    return make_toy_problem(
        IssueSchedule.daily(at="09:00", covers=("15h", "38h"), target_freq="1h")
    )
