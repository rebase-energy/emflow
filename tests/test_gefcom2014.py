"""GEFCom2014 benchmark: calendar/reference-score checks (no data needed) and,
when the local build cache is present, information-set and end-to-end checks."""

import numpy as np
import pandas as pd
import pytest

from emflow.benchmarks.gefcom2014.problem import (
    LOCAL_BUILD,
    QUANTILES,
    TRACKS,
    reference_scores,
    splits,
    task_origins,
)

HAS_DATA = (LOCAL_BUILD / "public" / "rebase.yaml").exists()


class TestCalendar:
    def test_origin_counts_per_track(self):
        assert len(task_origins("load")) == 15
        assert len(task_origins("price")) == 15
        assert len(task_origins("wind")) == 150   # 15 tasks x 10 zones
        assert len(task_origins("solar")) == 45   # 15 tasks x 3 zones

    def test_targets_follow_asof(self):
        for track in TRACKS:
            for o in task_origins(track):
                assert (o.target_index > o.asof).all()
                assert o.column is not None

    def test_splits_partition_tasks(self):
        for track in TRACKS:
            s = splits(track)
            origins = task_origins(track)
            val = [o for o in origins if o.target_start >= s.validation[0]
                   and o.target_start < s.validation[1]]
            hold = [o for o in origins if o.target_start >= s.holdout[0]
                    and o.target_start < s.holdout[1]]
            n_zones = len(TRACKS[track]["zones"])
            assert len(val) == 3 * n_zones
            assert len(hold) == 12 * n_zones


class TestReferenceScores:
    def test_official_winners(self):
        assert reference_scores("solar")[0].team == "Gang-gang"
        assert reference_scores("wind")[0].team == "kPower"
        assert reference_scores("load")[0].team == "Adada"
        for track in TRACKS:
            scores = [r.score for r in reference_scores(track)]
            assert scores == sorted(scores)  # lower pinball = better rank


@pytest.mark.skipif(not HAS_DATA, reason="gefcom2014 local build cache not present")
class TestInformationSet:
    def test_solar_nwp_knowable_only_from_its_task(self):
        """Task k's NWP for the target month must be invisible at task k-1."""
        import emflow as ef
        from emflow.data import DataFeed

        problem = ef.load_problem("gefcom2014:solar")
        feed = DataFeed(problem.load_dataset())
        origins = sorted({o.asof for o in problem.origins("holdout")})
        earlier, current = origins[0], origins[1]
        fc_now = feed.forecasts(current, "solar_nwp")
        month_targets = fc_now.index[fc_now.index > current]
        assert len(month_targets), "current task should see its target month NWP"
        fc_before = feed.forecasts(earlier, "solar_nwp")
        assert fc_before.index.max() < month_targets.max()

    def test_targets_release_follows_task_calendar(self):
        """Actuals for task k's month are knowable at task k+1, not at task k."""
        import emflow as ef
        from emflow.data import DataFeed

        problem = ef.load_problem("gefcom2014:wind")
        feed = DataFeed(problem.load_dataset())
        tasks = sorted({(o.asof, o.target_start, o.target_end)
                        for o in problem.origins("holdout")})
        asof_k, start_k, end_k = tasks[0]
        asof_next = tasks[1][0]
        at_k = feed.history(asof_k, "wind_targets")
        assert at_k.index.max() < start_k
        at_next = feed.history(asof_next, "wind_targets")
        assert at_next.index.max() >= end_k

    def test_load_track_has_no_future_temperature(self):
        import emflow as ef
        from emflow.data import DataFeed

        problem = ef.load_problem("gefcom2014:load")
        feed = DataFeed(problem.load_dataset())
        origin = problem.origins("holdout")[0]
        temp = feed.history(origin.asof, "load_temperature")
        assert temp.index.max() < origin.target_start


@pytest.mark.skipif(not HAS_DATA, reason="gefcom2014 local build cache not present")
class TestEndToEnd:
    def test_baseline_predicts_valid_quantiles_one_origin(self):
        import emflow as ef
        from emflow.benchmarks.gefcom2014.baseline import ClimatologyQuantiles

        problem = ef.load_problem("gefcom2014:solar")
        env = problem.env("validation")
        obs, info = env.reset()
        model = ClimatologyQuantiles().bind(problem)
        model.fit(info["train"])
        pred = model.predict(obs)
        assert list(pred.columns) == list(QUANTILES)
        assert pred.notna().all().all()
        vals = pred.to_numpy()
        assert (vals[:, :-1] <= vals[:, 1:] + 1e-12).all()  # non-crossing
