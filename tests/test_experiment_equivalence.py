"""The vectorized-mode correctness guarantee: event == vectorized, exactly."""

import numpy as np
import pandas as pd
import pytest

from emflow.data import DataPortal
from emflow.features import Calendar, ForecastField, Lag, Rolling, materialize
from emflow.features.materialize import materialize_observation, supervised_frame
from emflow.models.predictor import FeaturePredictor
from emflow.run import Experiment

from conftest import make_toy_problem


FEATURES = [
    Lag("power", ["1h", "2h", "24h"]),
    Rolling("power", "48h", "mean"),
    ForecastField("nwp", ["signal"]),
    Calendar(["hour", "dayofweek"]),
]


class LinearFeatureModel(FeaturePredictor):
    """OLS on the declarative features — batch-capable by construction."""

    features = tuple(FEATURES)

    def fit(self, train):
        problem = self._problem
        origins = problem.schedule.origins(
            train.history("power").index[200],  # skip warm-up so lags resolve
            train.asof,
        )
        portal = DataPortal(problem.load_dataset())
        X, y = supervised_frame(portal, self.features, origins, "power")
        mask = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
        A = np.column_stack([np.ones(mask.sum()), X.to_numpy()[mask]])
        self.beta, *_ = np.linalg.lstsq(A, y.to_numpy()[mask], rcond=None)
        return self

    def predict_tabular(self, X):
        with np.errstate(all="ignore"):  # NaN feature rows predict NaN, silently
            vals = self.beta[0] + X.to_numpy() @ self.beta[1:]
        return pd.DataFrame({"point": vals}, index=X.index)


@pytest.fixture
def fitted_setup():
    problem = make_toy_problem()
    model = LinearFeatureModel()
    model._problem = problem  # fit() needs the schedule to enumerate training origins
    return problem, model


class TestMaterializeEquivalence:
    def test_single_origin_matches_batch_rows(self, toy_problem):
        portal = DataPortal(toy_problem.load_dataset())
        origins = toy_problem.origins("validation")[:50]
        batch = materialize(portal, FEATURES, origins)
        for origin in origins[::7]:
            single = materialize_observation(portal.view(origin.asof), FEATURES,
                                             origin.target_index)
            batch_rows = batch.loc[[origin.asof]]
            pd.testing.assert_frame_equal(single, batch_rows, check_exact=False)

    def test_lag_masked_when_not_knowable(self, toy_problem):
        """A 30-minute lag on a field with 1h availability can never be known
        at the origin — every row must be NaN, not a peeked value."""
        portal = DataPortal(toy_problem.load_dataset())
        origins = toy_problem.origins("validation")[:10]
        X = materialize(portal, [Lag("power", ["30min"])], origins)
        assert X["power_lag_30min"].isna().all()


class TestModeEquivalence:
    def test_event_equals_vectorized(self, fitted_setup):
        problem, model = fitted_setup
        event = Experiment(problem, model.copy(), analyzers=None).run(mode="event")
        vect = Experiment(problem, model.copy(), analyzers=None).run(mode="vectorized")

        assert event.score == pytest.approx(vect.score, abs=1e-12)
        pd.testing.assert_frame_equal(event.predictions, vect.predictions)
        pd.testing.assert_frame_equal(event.settlements, vect.settlements)

    def test_auto_picks_vectorized_for_batch_models(self, fitted_setup):
        problem, model = fitted_setup
        result = Experiment(problem, model, analyzers=None).run(mode="auto")
        assert result.mode == "vectorized"

    def test_model_beats_persistence_sanity(self, fitted_setup):
        """The linear model knows the diurnal cycle; persistence doesn't."""
        problem, model = fitted_setup
        result = Experiment(problem, model).run()
        skill = result.analysis["PersistenceSkill"]
        assert skill["beats_persistence"]
