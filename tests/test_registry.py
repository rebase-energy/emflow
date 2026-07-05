import pandas as pd
import pytest

import emflow
from emflow.problems import Problem, ProblemNotIngestedError, Splits, list_problems, load_problem


class TestSplits:
    def test_overlap_with_training_rejected(self):
        with pytest.raises(ValueError, match="train_end"):
            Splits(train_end="2026-01-01",
                   validation=("2025-12-01", "2026-02-01"),
                   holdout=("2026-02-01", "2026-03-01"))

    def test_period_lookup(self):
        s = Splits(train_end="2026-01-01",
                   validation=("2026-01-01", "2026-02-01"),
                   holdout=("2026-02-01", "2026-03-01"))
        assert s.period("holdout")[0] == pd.Timestamp("2026-02-01")
        with pytest.raises(ValueError):
            s.period("test")


class TestRegistry:
    def test_all_registered_problems_load_or_declare_not_ingested(self):
        names = list_problems()
        assert names, "registry found no problems"
        for name in names:
            try:
                problem = load_problem(name)
            except ProblemNotIngestedError:
                continue  # registered, data pending — explicitly allowed
            assert isinstance(problem, Problem), f"{name} did not return a Problem"

    def test_unknown_problem_raises_keyerror(self):
        with pytest.raises(KeyError, match="unknown problem"):
            load_problem("no-such-problem")


class TestRank:
    def test_rank_of(self):
        from emflow.problems import MeanAbsoluteError, Objective, RefScore
        from emflow.problems.schedule import IssueSchedule

        problem = Problem(
            name="toy", dataset=lambda: None, make_env=lambda p, s: None,
            objective=Objective(MeanAbsoluteError()),
            schedule=IssueSchedule.hourly(),
            splits=Splits("2026-01-01", ("2026-01-01", "2026-02-01"),
                          ("2026-02-01", "2026-03-01")),
            reference_scores=[RefScore(1, "winner", 10.0), RefScore(2, "second", 12.0),
                              RefScore(3, "third", 15.0)],
        )
        assert problem.rank_of(9.0) == 1    # beats everyone
        assert problem.rank_of(11.0) == 2   # between 1st and 2nd
        assert problem.rank_of(20.0) == 4   # behind the field
