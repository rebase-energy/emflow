"""Truncation invariance: the strongest leak test.

Everything computed at origin *asof* must be bit-identical whether the dataset
physically contains the future or not. This is stronger than the equivalence
tests (which read the full frame on both sides — a *shared* masking bug would
pass them): here the future is deleted, so any dependence on it changes the
output.
"""

import numpy as np
import pandas as pd
import pytest

from emflow.data import DataPortal, Dataset, Field
from emflow.data.field import ISSUE_LEVEL
from emflow.features import Calendar, ForecastField, Lag, Rolling, materialize

from conftest import make_toy_problem
from test_portal import bitemporal_field

FEATURES = [
    Lag("power", ["1h", "2h", "24h"]),
    Rolling("power", "48h", "mean"),
    ForecastField("nwp", ["signal"], interp="time"),
    Calendar(["hour", "dayofweek"]),
]


def truncate_dataset(dataset: Dataset, asof: pd.Timestamp) -> Dataset:
    """Physically remove everything not knowable at ``asof`` from every field."""
    fields = {}
    for name, f in dataset.fields.items():
        if f.kind == "actual":
            if f.knowledge_col is not None:
                keep = f.frame[f.knowledge_col] <= asof
                frame = f.frame[keep.to_numpy()]
            else:
                frame = f.frame.loc[:asof - f.availability_lag]
            fields[name] = Field(name, frame.copy(), kind="actual",
                                 availability_lag=f.availability_lag,
                                 knowledge_col=f.knowledge_col)
        elif f.kind == "forecast":
            issues = f.frame.index.get_level_values(ISSUE_LEVEL)
            frame = f.frame[(issues + f.availability_lag) <= asof]
            fields[name] = Field(name, frame.copy(), kind="forecast",
                                 availability_lag=f.availability_lag)
        else:
            fields[name] = f
    return Dataset(name=dataset.name, fields=fields)


@pytest.fixture
def problem():
    return make_toy_problem()


def sample_origins(problem, k=6):
    origins = problem.origins("validation")
    return origins[:: max(1, len(origins) // k)]


class TestPortalInvariance:
    def test_history_and_forecasts(self, problem):
        full = DataPortal(problem.load_dataset())
        for origin in sample_origins(problem):
            trunc = DataPortal(truncate_dataset(problem.load_dataset(), origin.asof))
            pd.testing.assert_frame_equal(
                full.history(origin.asof, "power"),
                trunc.history(origin.asof, "power"))
            pd.testing.assert_frame_equal(
                full.forecasts(origin.asof, "nwp"),
                trunc.forecasts(origin.asof, "nwp"))

    def test_bitemporal_history(self):
        dataset = Dataset(name="d", fields={"power": bitemporal_field()})
        full = DataPortal(dataset)
        for asof in pd.date_range("2025-01-01 06:00", "2025-01-04", freq="11h", tz="UTC"):
            trunc = DataPortal(truncate_dataset(dataset, asof))
            pd.testing.assert_frame_equal(full.history(asof, "power"),
                                          trunc.history(asof, "power"))


class TestMaterializerInvariance:
    def test_all_spec_types(self, problem):
        full = DataPortal(problem.load_dataset())
        for origin in sample_origins(problem):
            trunc = DataPortal(truncate_dataset(problem.load_dataset(), origin.asof))
            pd.testing.assert_frame_equal(
                materialize(full, FEATURES, [origin]),
                materialize(trunc, FEATURES, [origin]))


class TestEndToEndInvariance:
    def test_fitted_model_predictions_identical(self, problem):
        from test_experiment_equivalence import LinearFeatureModel

        model = LinearFeatureModel()
        model._problem = problem
        env = problem.env("validation")
        _, info = env.reset()
        model.fit(info["train"])

        full = DataPortal(problem.load_dataset())
        for origin in sample_origins(problem, k=3):
            trunc = DataPortal(truncate_dataset(problem.load_dataset(), origin.asof))
            pred_full = model.predict_tabular(materialize(full, model.features, [origin]))
            pred_trunc = model.predict_tabular(materialize(trunc, model.features, [origin]))
            pd.testing.assert_frame_equal(pred_full, pred_trunc)


class TestStrictBoundary:
    def test_train_view_equals_strict_truncation(self, problem):
        """A strict view at train_end must equal an inclusive view over a
        dataset truncated one instant before train_end — guards the boundary
        fix where the value stamped exactly train_end is a scored target."""
        cutoff = problem.splits.train_end
        full = DataPortal(problem.load_dataset())
        trunc = DataPortal(truncate_dataset(problem.load_dataset(),
                                            cutoff - pd.Timedelta("1ns")))
        pd.testing.assert_frame_equal(
            full.history(cutoff, "power", strict=True),
            trunc.history(cutoff, "power", strict=False))
