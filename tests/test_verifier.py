"""Verifier policy + the anti-cheat guarantee.

The key tests: deliberately-cheating models must gain *nothing* — not merely
be flagged. A cheater that tries to read its targets from the training view or
the observation finds they simply aren't there.
"""

import numpy as np
import pandas as pd
import pytest

from emflow.models.predictor import Predictor
from emflow.run import Experiment
from emflow.run.verifier import Verifier

from conftest import make_toy_problem


class HonestPersistence(Predictor):
    def predict(self, obs):
        last = obs.history("power")["power"].dropna().iloc[-1]
        return pd.DataFrame({"point": last}, index=obs.target_index)


class TargetPeeker(Predictor):
    """Tries to read the actual value at each target time from the observation.

    If any peeked value comes back, it forecasts perfectly; otherwise it falls
    back to a terrible constant — so a nonzero advantage over the constant
    proves a leak."""

    def __init__(self):
        super().__init__()
        self.peeked = 0

    def predict(self, obs):
        hist = obs.history("power")["power"]
        vals = []
        for t in obs.target_index:
            if t in hist.index and pd.notna(hist.loc[t]):
                self.peeked += 1
                vals.append(hist.loc[t])       # the leak, if it existed
            else:
                vals.append(-9999.0)
        return pd.DataFrame({"point": vals}, index=obs.target_index)


class TrainOnTestCheater(Predictor):
    """Tries to memorize post-train_end values from the training view."""

    def __init__(self):
        super().__init__()
        self.memorized = {}

    def fit(self, train):
        hist = train.history("power")["power"]
        future = hist[hist.index >= train.asof]   # anything at/after the cutoff
        self.memorized = future.to_dict()
        return self

    def predict(self, obs):
        vals = [self.memorized.get(t, -9999.0) for t in obs.target_index]
        return pd.DataFrame({"point": vals}, index=obs.target_index)


class TestAntiCheat:
    def test_target_peeker_gains_nothing(self):
        problem = make_toy_problem()
        model = TargetPeeker()
        result = Experiment(problem, model, analyzers=None, split="holdout").run()
        assert model.peeked == 0, "observation served a target value — LEAK"
        assert result.score > 1000  # only the fallback constant was ever used

    def test_train_view_contains_no_test_data(self):
        problem = make_toy_problem()
        model = TrainOnTestCheater()
        result = Experiment(problem, model, analyzers=None, split="holdout").run()
        assert model.memorized == {}, "training view leaked post-cutoff data"
        assert result.score > 1000

    def test_honest_model_scores_normally(self):
        problem = make_toy_problem()
        result = Experiment(problem, HonestPersistence(), analyzers=None,
                            split="holdout").run()
        assert result.score < 5.0


class TestVerifierPolicy:
    def test_accepts_instance_and_factory(self, tmp_path):
        problem = make_toy_problem()
        lb = tmp_path / "leaderboard.csv"
        verifier = Verifier(problem, leaderboard_path=lb)
        r1 = verifier.verify(HonestPersistence(), name="instance", verbose=False)
        r2 = verifier.verify(lambda: HonestPersistence(), name="factory", verbose=False)
        assert r1.split == r2.split == "holdout"
        rows = pd.read_csv(lb)
        assert list(rows["submission"]) == ["instance", "factory"]
        assert (rows["problem"] == "toy:hourly").all()

    def test_rejects_non_predictor(self):
        verifier = Verifier(make_toy_problem(), leaderboard_path=None)
        with pytest.raises(TypeError, match="Predictor"):
            verifier.verify(object())

    def test_scores_holdout_not_validation(self):
        problem = make_toy_problem()
        verifier = Verifier(problem, leaderboard_path=None)
        result = verifier.verify(HonestPersistence(), verbose=False)
        holdout_start, holdout_end = problem.splits.period("holdout")
        scored = result.predictions.index.get_level_values("target_time")
        assert scored.min() >= holdout_start
        assert scored.max() <= holdout_end
