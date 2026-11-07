"""Problem discovery and loading.

Problems live in packages under ``emflow.benchmarks`` (competition problems —
the product) and ``emflow.examples`` (tutorials), and any installed
distribution can register more through the ``emflow.problems`` entry-point
group. A package participates by providing a ``problem`` module with either

* ``list_problem_variants() -> list[str]`` + one ``get_problem_<variant>()``
  factory per variant, or
* a single ``get_problem()`` factory.

Factories must return a :class:`~emflow.problems.problem.Problem`. A benchmark
whose data has not been ingested yet raises :class:`ProblemNotIngestedError`
from its factory — it still shows up in :func:`list_problems` (marked in
:func:`problem_status`), so the full roster is visible.

Packages whose name starts with ``_`` (e.g. ``examples/_legacy``) are skipped.

External problems
-----------------
A project that keeps its own problems (private data, internal splits) registers
them in its ``pyproject.toml`` and they load like any packaged one::

    [project.entry-points."emflow.problems"]
    icing = "my_signals.icing.problem"

    >>> emflow.load_problem("icing:se1")

The entry-point name is the problem's folder name (the part before ``:``) and
the target is a module following the ``problem`` module convention above.
Packaged problems win on a name collision.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
import pkgutil
import typing as t
import warnings

from .problem import Problem

BASES = ("emflow.benchmarks", "emflow.examples")
ENTRY_POINT_GROUP = "emflow.problems"


class ProblemNotIngestedError(NotImplementedError):
    """Raised by a registered benchmark whose dataset is not yet ingested."""


def _packaged_modules():
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


def _external_modules(name: t.Optional[str] = None):
    """Problem modules registered by other distributions (``emflow.problems``).

    A registration that fails to import is skipped with a warning rather than
    taking the whole registry down with it.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if name is not None and ep.name != name:
            continue
        if ":" in ep.name:
            warnings.warn(f"emflow.problems entry point {ep.name!r} skipped: names cannot contain ':'")
            continue
        try:
            yield ep.name, ep.load()
        except Exception as exc:  # noqa: BLE001 — one bad plugin must not hide the rest
            warnings.warn(f"emflow.problems entry point {ep.name!r} failed to load: {exc!r}")


def _iter_problem_modules():
    seen = set()
    for name, module in _packaged_modules():
        seen.add(name)
        yield name, module
    for name, module in _external_modules():
        if name not in seen:
            yield name, module


def list_problems() -> t.List[str]:
    """All registered problem names (``package`` or ``package:variant``)."""
    problems = []
    for name, module in _iter_problem_modules():
        if hasattr(module, "list_problem_variants"):
            problems.extend(f"{name}:{v}" for v in module.list_problem_variants())
        elif hasattr(module, "get_problem"):
            problems.append(name)
    return sorted(problems)


def _problem_module(name: str):
    folder = name.partition(":")[0]
    modname = folder.replace("-", "_")
    for base in BASES:
        try:
            return importlib.import_module(f"{base}.{modname}.problem")
        except ModuleNotFoundError:
            continue
    for _, module in _external_modules(folder):
        return module
    raise KeyError(f"unknown problem {name!r}; known: {list_problems()}")


def cache_problem_data(name: str, include_private: bool = False, token: str | None = None) -> bool:
    """Materialize a benchmark's HF-hosted data into its local build cache so
    later loads (e.g. sandboxed evaluations running offline and without
    credentials) read from disk.

    Benchmarks that follow the ``PUBLIC_REPO``/``PRIVATE_REPO``/``LOCAL_BUILD``
    convention are downloaded via huggingface_hub (token from the argument or
    ambient env). Problems that ship packaged data have nothing to cache —
    returns False. ``include_private`` deliberately defaults to False: holdout
    labels should stay off local disk in sandboxed setups and be read straight
    from the gated repo by the credentialed verifier instead.
    """
    module = _problem_module(name)
    build = getattr(module, "LOCAL_BUILD", None)
    public = getattr(module, "PUBLIC_REPO", None)
    if build is None or public is None:
        return False
    from huggingface_hub import snapshot_download

    repos = [(public, "public")]
    if include_private and getattr(module, "PRIVATE_REPO", None):
        repos.append((module.PRIVATE_REPO, "private"))
    for repo, kind in repos:
        repo_id = repo.removeprefix("rb://dataset/")
        target = build / kind
        if (target / "rebase.yaml").exists():
            continue
        snapshot_download(repo_id=repo_id, repo_type="dataset",
                          local_dir=str(target), token=token)
    return True


def load_problem(name: str) -> Problem:
    """Load a problem by name; always returns a :class:`Problem`.

    Raises :class:`ProblemNotIngestedError` for registered-but-not-ingested
    benchmarks, ``KeyError`` for unknown names.
    """
    folder, _, variant = name.partition(":")
    module = _problem_module(name)

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
