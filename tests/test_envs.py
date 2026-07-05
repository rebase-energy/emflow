import numpy as np
import pandas as pd
import pytest

from emflow.envs import ForecastEnv, Observation


def naive_action(obs) -> pd.DataFrame:
    last = obs.history("power")["power"].dropna().iloc[-1]
    return pd.DataFrame({"point": last}, index=obs.target_index)


class TestGymContract:
    def test_reset_returns_obs_and_train_view(self, toy_problem):
        env = toy_problem.env("validation")
        obs, info = env.reset()
        assert isinstance(obs, Observation)
        assert "train" in info
        assert info["train"].asof == toy_problem.splits.train_end

    def test_step_five_tuple_and_termination(self, toy_problem):
        env = toy_problem.env("validation")
        obs, _ = env.reset()
        n = len(env.origins)
        steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(naive_action(obs))
            assert isinstance(reward, float) and isinstance(info["settled"], list)
            steps += 1
            if terminated:
                break
        assert steps == n
        assert obs is None

    def test_observation_cannot_see_targets(self, toy_problem):
        """The information set at any origin excludes every target timestamp."""
        env = toy_problem.env("validation")
        obs, _ = env.reset()
        while obs is not None:
            hist = obs.history("power")
            assert hist.index.max() < obs.target_index[0]
            obs, *_ , info = env.step(naive_action(obs))

    def test_all_origins_settled_by_the_end(self, toy_problem):
        env = toy_problem.env("validation")
        obs, _ = env.reset()
        settled = 0
        while True:
            obs, _, terminated, _, info = env.step(naive_action(obs))
            settled += len(info["settled"])
            if terminated:
                break
        assert settled == len(env.origins)


class TestDelayedReward:
    def test_dayahead_scores_arrive_only_after_actuals(self, toy_dayahead_problem):
        """A day-ahead origin (covering up to asof+38h) cannot settle at the
        next day's origin (asof+24h) — its window hasn't closed yet."""
        env = toy_dayahead_problem.env("validation")
        obs, _ = env.reset()
        obs, reward, _, _, info = env.step(naive_action(obs))
        assert info["settled"] == [] and reward == 0.0
        # After the second step, the first origin's window (ending asof+38h)
        # is still not fully knowable at the third origin (asof+48h > 38h+1h ✓ it is)
        obs, reward, _, _, info = env.step(naive_action(obs))
        assert len(info["settled"]) == 1

    def test_reward_sign_lower_is_better(self, toy_problem):
        env = toy_problem.env("validation")
        obs, _ = env.reset()
        total = 0.0
        while True:
            obs, reward, terminated, _, info = env.step(naive_action(obs))
            total += reward
            if terminated:
                break
        assert total < 0  # negated MAE


class TestActionValidation:
    def test_missing_target_times_rejected(self, toy_problem):
        env = toy_problem.env("validation")
        obs, _ = env.reset()
        bad = pd.DataFrame({"point": [0.0]}, index=obs.target_index[:1][::-1][:0])
        with pytest.raises(ValueError, match="missing forecasts"):
            env.step(pd.DataFrame({"point": []}, index=pd.DatetimeIndex([], tz="UTC")))

    def test_non_dataframe_rejected(self, toy_problem):
        env = toy_problem.env("validation")
        env.reset()
        with pytest.raises(TypeError, match="DataFrame"):
            env.step(np.zeros(3))

    def test_single_column_coerced_to_point(self, toy_problem):
        env = toy_problem.env("validation")
        obs, _ = env.reset()
        action = pd.DataFrame({"whatever": 1.0}, index=obs.target_index)
        _, _, _, _, info = env.step(action)  # no error

    def test_quantile_contract_enforced(self, toy_problem):
        problem = toy_problem
        env = ForecastEnv(
            feed=problem.env("validation").feed,
            origins=problem.origins("validation"),
            target_field="power",
            objective=problem.objective,
            quantiles=[0.1, 0.5, 0.9],
        )
        obs, _ = env.reset()
        with pytest.raises(ValueError, match="quantile columns"):
            env.step(pd.DataFrame({"point": 1.0}, index=obs.target_index))
        ok = pd.DataFrame({0.1: 0.0, 0.5: 1.0, 0.9: 2.0}, index=obs.target_index)
        env.step(ok)
