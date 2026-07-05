"""HEFTCom2024 benchmark: schedule correctness (no data needed) and, when the
local build cache is present, end-to-end problem loading."""

from pathlib import Path

import pandas as pd
import pytest

from emflow.benchmarks.heftcom2024.problem import (
    LOCAL_BUILD,
    market_day_origins,
    reference_scores,
)


class TestMarketDaySchedule:
    def test_unique_asofs_across_dst(self):
        origins = market_day_origins(pd.Timestamp("2024-03-25", tz="UTC"),
                                     pd.Timestamp("2024-04-05", tz="UTC"))
        asofs = [o.asof for o in origins]
        assert len(asofs) == len(set(asofs)), "DST collapsed two market days onto one asof"

    def test_spring_forward_day_has_46_periods(self):
        origins = market_day_origins(pd.Timestamp("2024-03-30", tz="UTC"),
                                     pd.Timestamp("2024-04-02", tz="UTC"))
        by_day = {o.target_index[0].tz_convert("Europe/Paris").date(): len(o.target_index)
                  for o in origins}
        import datetime
        assert by_day[datetime.date(2024, 3, 31)] == 46  # 23h CET->CEST day
        assert by_day[datetime.date(2024, 4, 1)] == 48

    def test_blocks_match_official_trades_archive(self):
        """First scored block 2024-02-19 23:00 UTC; DST block ends 21:30 UTC."""
        origins = {o.target_index[0]: o for o in market_day_origins()}
        first = origins[pd.Timestamp("2024-02-19 23:00", tz="UTC")]
        assert len(first.target_index) == 48
        assert first.target_index[-1] == pd.Timestamp("2024-02-20 22:30", tz="UTC")
        dst = origins[pd.Timestamp("2024-03-30 23:00", tz="UTC")]
        assert len(dst.target_index) == 46
        assert dst.target_index[-1] == pd.Timestamp("2024-03-31 21:30", tz="UTC")

    def test_asof_precedes_all_targets(self):
        for o in market_day_origins(pd.Timestamp("2024-02-01", tz="UTC"),
                                    pd.Timestamp("2024-02-10", tz="UTC")):
            assert (o.target_index > o.asof).all()
            assert o.asof.hour == 9 and o.asof.minute == 20

    def test_targets_are_half_hourly_utc(self):
        (o, *_) = market_day_origins(pd.Timestamp("2024-02-01", tz="UTC"),
                                     pd.Timestamp("2024-02-03", tz="UTC"))
        deltas = o.target_index.to_series().diff().dropna().unique()
        assert list(deltas) == [pd.Timedelta("30min")]


class TestReferenceScores:
    def test_forecasting_leaderboard_packaged(self):
        scores = reference_scores("forecasting")
        assert scores[0].team == "SVK"
        assert scores[0].score == pytest.approx(22.18, abs=0.01)
        assert len(scores) >= 20


RAW_TRADES = LOCAL_BUILD.parent / "raw" / "trades.csv"


@pytest.mark.skipif(not ((LOCAL_BUILD / "public" / "rebase.yaml").exists()
                         and RAW_TRADES.exists()),
                    reason="heftcom2024 build cache / trades archive not present")
class TestSettlementGolden:
    def test_replaying_winner_bids_reproduces_official_revenue(self):
        """The strongest settlement check: SVK's actual bids through our
        TradingEnv must reproduce their official total revenue (small residual
        = data revisions between archive snapshots)."""
        import emflow as ef
        from emflow.models.predictor import Predictor

        trades = pd.read_csv(RAW_TRADES, parse_dates=["dtm"], low_memory=False)
        svk = trades[trades.team == "SVK"].set_index("dtm").sort_index()

        class SVKReplay(Predictor):
            def predict(self, obs):
                return pd.DataFrame({"bid": svk["market_bid"].reindex(obs.target_index)},
                                    index=obs.target_index)

        problem = ef.load_problem("heftcom2024:trading")
        result = ef.Experiment(problem, SVKReplay(), analyzers=None, split="holdout").run()
        official = svk["revenue"].sum()
        assert abs(result.score - official) / official < 1e-4


@pytest.mark.skipif(not (LOCAL_BUILD / "public" / "rebase.yaml").exists(),
                    reason="heftcom2024 local build cache not present")
class TestEndToEnd:
    def test_problem_loads_and_baseline_predicts_one_origin(self):
        import emflow as ef
        from emflow.benchmarks.heftcom2024.baseline import BinnedQuantileBaseline

        problem = ef.load_problem("heftcom2024:forecasting")
        env = problem.env("validation")
        obs, info = env.reset()
        model = BinnedQuantileBaseline(train_window="90D").bind(problem)
        model.fit(info["train"])
        pred = model.predict(obs)
        assert list(pred.columns) == [q / 10 for q in range(1, 10)]
        assert pred.notna().all().all()
        assert (pred.to_numpy()[:, :-1] <= pred.to_numpy()[:, 1:]).all()  # non-crossing
