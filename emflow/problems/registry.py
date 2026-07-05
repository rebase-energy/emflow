"""Problem discovery and loading.

Problems live in packages under ``emflow.benchmarks`` (competition problems —
the product) and ``emflow.examples`` (tutorials). A package participates by
providing a ``problem`` module with either

* ``list_problem_variants() -> list[str]`` + one ``get_problem_<variant>()``
  factory per variant, or
* a single ``get_problem()`` factory.

Factories must return a :class:`~emflow.problems.problem.Problem`. A benchmark
whose data has not been ingested yet raises :class:`ProblemNotIngestedError`
from its factory — it still shows up in :func:`list_problems` (marked in
:func:`problem_status`), so the full roster is visible.

Packages whose name starts with ``_`` (e.g. ``examples/_legacy``) are skipped.
"""

from __future__ import annotations

import importlib
import pkgutil
import typing as t

from .problem import Problem

BASES = ("emflow.benchmarks", "emflow.examples")


class ProblemNotIngestedError(NotImplementedError):
    """Raised by a registered benchmark whose dataset is not yet ingested."""


def _iter_problem_modules():
    for base in BASES:
        try:
            pkg = importlib.import_module(base)
        except ModuleNotFoundError:
            continue
        for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
            if not ispkg or modname.startswith("_") or modname == "data":
                continue
            try:
                module = importlib.import_module(f"{base}.{modname}.problem")
            except ModuleNotFoundError:
                continue
            yield modname.replace("_", "-"), module


def list_problems() -> t.List[str]:
    """All registered problem names (``package`` or ``package:variant``)."""
    problems = []
    for name, module in _iter_problem_modules():
        if hasattr(module, "list_problem_variants"):
            problems.extend(f"{name}:{v}" for v in module.list_problem_variants())
        elif hasattr(module, "get_problem"):
            problems.append(name)
    return sorted(problems)


def load_problem(name: str) -> Problem:
    """Load a problem by name; always returns a :class:`Problem`.

    Raises :class:`ProblemNotIngestedError` for registered-but-not-ingested
    benchmarks, ``KeyError`` for unknown names.
    """
    folder, _, variant = name.partition(":")
    modname = folder.replace("-", "_")

    module = None
    for base in BASES:
        try:
            module = importlib.import_module(f"{base}.{modname}.problem")
            break
        except ModuleNotFoundError:
            continue
    if module is None:
        raise KeyError(f"unknown problem {name!r}; known: {list_problems()}")

    factory_name = f"get_problem_{variant}" if variant else "get_problem"
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise KeyError(
            f"problem package {folder!r} has no factory {factory_name!r}; "
            f"known problems: {list_problems()}"
        )

    problem = factory()
    if not isinstance(problem, Problem):
        raise TypeError(
            f"factory for {name!r} returned {type(problem).__name__}, expected Problem — "
            f"problem factories must return the Problem bundle"
        )
    return problem
