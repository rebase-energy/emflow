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


class TestExternalRegistration:
    """Problems registered by other distributions through ``emflow.problems``."""

    @staticmethod
    def _toy_problem(name):
        from emflow.problems import MeanAbsoluteError, Objective
        from emflow.problems.schedule import IssueSchedule

        return Problem(
            name=name, dataset=lambda: None, make_env=lambda p, s: None,
            objective=Objective(MeanAbsoluteError()), schedule=IssueSchedule.hourly(),
            splits=Splits("2026-01-01", ("2026-01-01", "2026-02-01"), ("2026-02-01", "2026-03-01")),
        )

    def _install(self, monkeypatch, registrations):
        from emflow.problems import registry

        class FakeEntryPoint:
            def __init__(self, name, module):
                self.name, self._module = name, module

            def load(self):
                if isinstance(self._module, Exception):
                    raise self._module
                return self._module

        eps = [FakeEntryPoint(n, m) for n, m in registrations]
        monkeypatch.setattr(registry, "entry_points", lambda group: eps if group == "emflow.problems" else [])

    def test_entry_point_problem_lists_and_loads(self, monkeypatch):
        import types

        module = types.ModuleType("acme_problems")
        module.list_problem_variants = lambda: ["se1", "se2"]
        module.get_problem_se1 = lambda: self._toy_problem("icing:se1")
        module.get_problem_se2 = lambda: self._toy_problem("icing:se2")
        self._install(monkeypatch, [("icing", module)])

        assert {"icing:se1", "icing:se2"} <= set(list_problems())
        assert load_problem("icing:se1").name == "icing:se1"
        with pytest.raises(KeyError, match="no factory"):
            load_problem("icing:se3")

    def test_packaged_problem_wins_name_collision(self, monkeypatch):
        import types

        module = types.ModuleType("shadow")
        module.get_problem = lambda: self._toy_problem("shadow")
        self._install(monkeypatch, [("swedish-temperatures", module)])

        names = list_problems()
        assert "swedish-temperatures:ar" in names
        assert "swedish-temperatures" not in names
        assert load_problem("swedish-temperatures:ar").name == "swedish-temperatures:ar"

    def test_broken_entry_point_is_skipped_with_warning(self, monkeypatch):
        import types

        good = types.ModuleType("good")
        good.get_problem = lambda: self._toy_problem("good")
        self._install(monkeypatch, [("broken", ModuleNotFoundError("nope")), ("good", good)])

        with pytest.warns(UserWarning, match="'broken' failed to load"):
            names = list_problems()
        assert "good" in names
        assert "broken" not in names
