"""Submission-module convention and the one-call evaluate helper.

This module is the stable integration surface for external drivers (e.g.
rebase-hillclimb's evaluator, the Rebase toolkit): a *submission* is a Python
file exposing a Predictor, loaded with :func:`load_submission`, and scored
with :func:`evaluate`. Treat signature changes here as breaking.

SECURITY: :func:`load_submission` imports and executes the file. Only load
code you trust, or run inside a sandboxed environment.
"""

from __future__ import annotations

import importlib.util
import typing as t
from pathlib import Path

from .experiment import Experiment
from .result import Result

if t.TYPE_CHECKING:
    from emflow.models.predictor import Predictor
    from emflow.problems.problem import Problem


def load_submission(path: str | Path) -> "Predictor":
    """Import a submission module and return the submitted Predictor.

    Accepts either form — whichever the submitter prefers:
      * a ``get_model() -> emflow.Predictor`` factory returning a fresh,
        untrained model (preferred; back-compatible with the original
        template), or
      * a module-level ``emflow.Predictor`` instance — one named ``model``,
        else the sole Predictor instance in the module.

    Either way the caller trains the model itself on the official split, so
    the leakage guarantees are identical.
    """
    from emflow.models.predictor import Predictor

    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import submission module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_model"):
        return module.get_model()

    candidate = getattr(module, "model", None)
    if isinstance(candidate, Predictor):
        return candidate
    instances = [v for v in vars(module).values() if isinstance(v, Predictor)]
    if len(instances) == 1:
        return instances[0]
    if len(instances) > 1:
        raise AttributeError(
            f"{path} defines several emflow.Predictor instances — name the one to "
            f"submit as `model = ...`, or expose get_model()."
        )
    raise AttributeError(
        f"{path} must define `model = <emflow.Predictor>` or "
        f"get_model() -> emflow.Predictor"
    )


def evaluate(
    problem: "str | Problem",
    model: "Predictor | str | Path",
    split: str = "validation",
    mode: str = "auto",
    analyzers: t.Union[str, t.Sequence, None] = "default",
) -> Result:
    """Fit + score a model on one split of a problem; returns the Result.

    ``problem`` may be a registry name (``"gefcom2014:solar"``) or a Problem.
    ``model`` may be a Predictor instance or a path to a submission file
    (loaded via :func:`load_submission`). The model is always (re)fit on the
    problem's official training view inside the Experiment loop.
    """
    if isinstance(problem, str):
        from emflow.problems.registry import load_problem

        problem = load_problem(problem)
    if isinstance(model, (str, Path)):
        model = load_submission(model)
    return Experiment(problem, model, analyzers=analyzers, split=split).run(mode=mode)
