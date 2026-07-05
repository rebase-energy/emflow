"""Transfer sweep — offline, on a seeded grid cache."""

import pandas as pd
import pytest

from emflow.run.transfer import best_per_problem, sweep
from tests.test_grid_io import _seed_cache


@pytest.fixture
def seeded_problem(monkeypatch, tmp_path):
    from emflow.benchmarks.grid import problem as grid_problem

    monkeypatch.setenv("EMFLOW_GRID_OFFLINE", "1")
    _seed_cache(monkeypatch, tmp_path, "DE-LU", "demand",
                grid_problem.DATA_END - grid_problem.HISTORY,
                grid_problem.DATA_END)
    return "grid:demand-de-lu"


@pytest.fixture
def baseline_solution(tmp_path):
    path = tmp_path / "baseline_solution.py"
    path.write_text("from emflow.benchmarks.grid.baseline import get_model\n")
    return path


def test_sweep_scores_and_records_errors(seeded_problem, baseline_solution, tmp_path):
    broken = tmp_path / "broken_solution.py"
    broken.write_text("raise RuntimeError('boom on import')\n")

    table = sweep(
        {"baseline": baseline_solution, "broken": broken},
        [seeded_problem, "grid:demand-fi"],  # fi not cached -> problem error
    )
    assert len(table) == 4

    ok = table[(table.solution == "baseline") & (table.problem == seeded_problem)]
    assert ok.score.iloc[0] == pytest.approx(0.0, abs=1e-9)  # pure sine: exact
    assert pd.isna(ok.error.iloc[0])

    failed = table[table.solution == "broken"]
    assert failed.score.isna().all()
    assert failed.error.str.contains("boom|problem").all()

    uncached = table[table.problem == "grid:demand-fi"]
    assert uncached.score.isna().all()


def test_best_per_problem_picks_winner(seeded_problem, baseline_solution):
    table = sweep({"a": baseline_solution, "b": baseline_solution}, [seeded_problem])
    best = best_per_problem(table)
    assert len(best) == 1
    assert best.problem.iloc[0] == seeded_problem
    assert best.score.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_public_api_exports():
    import emflow as ef

    assert ef.sweep is sweep
    assert ef.best_per_problem is best_per_problem
