"""Transfer evaluation: score solution modules across many problems.

A model class that wins a hillclimb search on one problem often works on the
sibling problems it was never searched on (the same variable in another
bidding zone, say). Searching every problem is expensive; *transferring* —
re-fitting each winning solution on every sibling's training split and
scoring it — costs seconds per pair and no agents. :func:`sweep` runs the
(solutions × problems) matrix and :func:`best_per_problem` picks the winner
per problem, giving full coverage from a handful of searched anchors.

Solutions are given as paths to submission modules (see
:mod:`emflow.run.submission`); each (solution, problem) cell reloads the
module so no fitted state leaks between evaluations. Failures — a variant
whose data is not ingested, a model incompatible with a problem — are
recorded in the ``error`` column rather than aborting the sweep.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import pandas as pd

from ..problems.registry import load_problem
from .submission import evaluate

Log = t.Callable[[str], None]


def sweep(
    solutions: t.Mapping[str, t.Union[str, Path]],
    problems: t.Iterable[str],
    split: str = "validation",
    log: t.Optional[Log] = None,
) -> pd.DataFrame:
    """Evaluate every solution module on every problem.

    Returns a tidy frame with one row per (solution, problem):
    ``solution, problem, score, objective, lower_is_better, error``.
    Failed cells carry ``score=NaN`` and the error message.
    """
    rows = []
    for problem_name in problems:
        try:
            problem = load_problem(problem_name)
            problem.load_dataset()
        except Exception as exc:  # noqa: BLE001 — incl. ProblemNotIngestedError
            for sol_name in solutions:
                rows.append(_row(sol_name, problem_name, error=f"problem: {exc}"))
            if log:
                log(f"{problem_name}: unavailable ({exc})")
            continue
        for sol_name, sol_path in solutions.items():
            try:
                result = evaluate(problem, sol_path, split=split)
                rows.append(_row(
                    sol_name, problem_name, score=result.score,
                    objective=result.objective,
                    lower_is_better=result.lower_is_better,
                ))
                if log:
                    log(f"{problem_name} × {sol_name}: {result.score:.4f}")
            except Exception as exc:  # noqa: BLE001 — record, keep sweeping
                rows.append(_row(sol_name, problem_name, error=str(exc)[:200]))
                if log:
                    log(f"{problem_name} × {sol_name}: FAILED ({str(exc)[:80]})")
    return pd.DataFrame(rows)


def best_per_problem(table: pd.DataFrame) -> pd.DataFrame:
    """The winning row per problem from a :func:`sweep` table.

    Respects each problem's ``lower_is_better``; problems where every cell
    failed are omitted.
    """
    scored = table.dropna(subset=["score"])
    winners = []
    for problem_name, group in scored.groupby("problem", sort=True):
        lower = bool(group["lower_is_better"].iloc[0])
        idx = group["score"].idxmin() if lower else group["score"].idxmax()
        winners.append(group.loc[idx])
    return pd.DataFrame(winners).reset_index(drop=True)


def _row(solution: str, problem: str, score: float = float("nan"),
         objective: t.Optional[str] = None,
         lower_is_better: t.Optional[bool] = None,
         error: t.Optional[str] = None) -> dict:
    return {
        "solution": solution, "problem": problem, "score": score,
        "objective": objective, "lower_is_better": lower_is_better,
        "error": error,
    }
