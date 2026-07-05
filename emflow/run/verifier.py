"""Verifier: Experiment + untrusted-submission policy.

The evaluation loop itself is the ordinary :class:`~emflow.run.experiment.Experiment`
— leakage protection lives in the data layer (every observation is a
portal-served view), not in a special runner. What the Verifier adds is
submission hygiene for *untrusted* models:

* the submitter provides a **fresh, untrained** :class:`~emflow.models.predictor.Predictor`
  (directly or via ``get_model()``); the verifier fits it itself on the
  official training view, so it can never have been fit on test targets;
* scoring runs on the **holdout** split, which agents must never iterate on
  (they hillclimb on ``validation``);
* the score is computed by the verifier from the returned predictions — nothing
  the submission prints is trusted;
* results append to a leaderboard CSV, and when the problem carries published
  reference scores the scorecard reports where the submission would have
  placed in the original competition.

Honest scope note: for problems whose full dataset ships locally (the tutorial
problems), a determined adversary can read the raw files outside the portal —
local verification protects against *accidental* leakage. The trustworthy
setup for agent benchmarking keeps holdout labels in a private ``rb://`` repo
that only the verifier's token can read, and runs submissions with no network.
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
import typing as t

from ..models.predictor import Predictor
from ..problems import Problem, load_problem
from .experiment import Experiment
from .result import Result

DEFAULT_LEADERBOARD = Path(__file__).resolve().parents[2] / "submissions" / "leaderboard.csv"

LEADERBOARD_FIELDS = [
    "submitted_at", "submission", "problem", "split", "objective", "score",
    "n_origins", "n_scored", "persistence_score", "beats_persistence", "rank",
]


class Verifier:
    """Train and strictly evaluate a submitted predictor on a problem's holdout."""

    def __init__(self, problem: t.Union[str, Problem] = "swedish-temperatures:ar",
                 leaderboard_path=DEFAULT_LEADERBOARD, split: str = "holdout"):
        self.problem = load_problem(problem) if isinstance(problem, str) else problem
        self.leaderboard_path = Path(leaderboard_path) if leaderboard_path else None
        self.split = split

    def verify(self, submission, name=None, record=True, verbose=True,
               mode: str = "auto", metadata: t.Optional[dict] = None) -> Result:
        """Evaluate a submission and return its :class:`Result`.

        ``submission`` is a fresh :class:`Predictor` or a zero-arg factory
        returning one (the ``get_model()`` convention).

        ``metadata`` (optional, str/number values) is recorded verbatim on the
        leaderboard row and echoed on the scorecard. This is the seam for
        search harnesses to make performance claims honest about selection:
        rebase-hillclimb passes e.g. ``{"n_trials": N}`` from its run journal,
        and any deflation/PBO math lives there — emflow only records it.
        """
        if metadata:
            collisions = set(metadata) & set(LEADERBOARD_FIELDS)
            if collisions:
                raise ValueError(
                    f"metadata keys {sorted(collisions)} collide with leaderboard fields"
                )
        model = submission() if callable(submission) and not isinstance(submission, Predictor) \
            else submission
        if not isinstance(model, Predictor):
            raise TypeError(
                f"submission must be an emflow Predictor (or a factory returning "
                f"one), got {type(model).__name__}"
            )
        if name:
            model.name = name

        result = Experiment(self.problem, model, split=self.split).run(mode=mode)

        if verbose:
            self._print_scorecard(result, metadata)
        if record and self.leaderboard_path:
            self._append_leaderboard(result, metadata)
        return result

    # -- reporting ----------------------------------------------------------------

    def _row(self, r: Result, metadata: t.Optional[dict] = None) -> dict:
        skill = r.analysis.get("PersistenceSkill", {})
        row = {
            "submitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "submission": r.model,
            "problem": r.problem,
            "split": r.split,
            "objective": r.objective,
            "score": round(r.score, 6),
            "n_origins": r.n_origins,
            "n_scored": r.n_scored,
            "persistence_score": round(skill.get("persistence_score", float("nan")), 6),
            "beats_persistence": skill.get("beats_persistence", ""),
            "rank": r.rank_against(self.problem) or "",
        }
        if metadata:
            row.update({key: metadata[key] for key in sorted(metadata)})
        return row

    def _print_scorecard(self, r: Result, metadata: t.Optional[dict] = None) -> None:
        row = self._row(r, metadata)
        print(f"\n{'=' * 64}")
        print(f"Submission : {r.model}")
        print(f"Problem    : {r.problem}  [{r.split} split]")
        print(f"Scored on  : {r.n_scored:,} timestamps over {r.n_origins:,} origins")
        print(f"{'-' * 64}")
        print(f"{r.objective:<32}{r.score:>12.4f}")
        skill = r.analysis.get("PersistenceSkill", {})
        if skill:
            print(f"{'Persistence baseline':<32}{skill['persistence_score']:>12.4f}")
        for key, value in r.analysis.get("QuantileCalibration", {}).items():
            print(f"{key:<32}{value:>12.3f}")
        print(f"{'-' * 64}")
        if skill:
            verdict = ("PASS — beats persistence" if skill["beats_persistence"]
                       else "FAIL — does not beat persistence")
            print(f"Verdict    : {verdict}")
        if row["rank"]:
            n = len(self.problem.reference_scores)
            print(f"Ranking    : would have placed {row['rank']} of {n} "
                  f"in the original competition")
        if metadata:
            for key in sorted(metadata):
                print(f"{key:<11}: {metadata[key]}")
        print(f"Leakage    : impossible by construction (portal-served observations)")
        print(f"{'=' * 64}\n")

    def _append_leaderboard(self, r: Result, metadata: t.Optional[dict] = None) -> None:
        path = self.leaderboard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        row = self._row(r, metadata)

        if path.exists():
            # Respect the existing header: fill absent columns, warn on and
            # drop keys the file doesn't know about (schema is fixed at creation).
            with path.open(newline="") as f:
                fieldnames = next(csv.reader(f))
            dropped = set(row) - set(fieldnames)
            if dropped:
                import warnings
                warnings.warn(
                    f"leaderboard {path} has no columns for metadata keys "
                    f"{sorted(dropped)}; they were not recorded"
                )
            row = {k: row.get(k, "") for k in fieldnames}
            new = False
        else:
            fieldnames = LEADERBOARD_FIELDS + sorted(set(row) - set(LEADERBOARD_FIELDS))
            new = True

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended to leaderboard: {path}")
