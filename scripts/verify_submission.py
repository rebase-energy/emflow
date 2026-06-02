"""Verify a submitted forecasting model and score it on an emflow problem.

Usage:
    python scripts/verify_submission.py <path/to/submission.py> \
        [--problem swedish-temperatures:ar] [--name NAME] \
        [--leaderboard submissions/leaderboard.csv] [--lookback HOURS]

The submission file must expose ``get_model() -> emflow.Predictor`` returning a
*fresh, untrained* predictor. The verifier trains it on the official training split
and scores it via strict walk-forward (the submitter never sees the test targets), so
the reported number cannot be inflated by data leakage.

SECURITY: this imports and executes the submission file. Only run code from people
you trust.
"""

import argparse
import importlib.util
from pathlib import Path

from emflow.experiments.verifier import Verifier, DEFAULT_LEADERBOARD


def load_submission(path: Path):
    """Import a submission module from a file path and return its get_model()."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_model"):
        raise AttributeError(
            f"{path} must define get_model() -> emflow.Predictor (returning a fresh model)"
        )
    return module.get_model()


def main():
    ap = argparse.ArgumentParser(description="Verify a submitted Predictor model.")
    ap.add_argument("submission", type=Path, help="path to the submission .py file")
    ap.add_argument("--problem", default="swedish-temperatures:ar")
    ap.add_argument("--name", default=None, help="submission name for the scorecard/leaderboard")
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    ap.add_argument("--lookback", type=int, default=None,
                    help="trailing hours of history per step (default: full history)")
    args = ap.parse_args()

    print(f"⚠  Importing and running {args.submission} — only do this for trusted code.")
    model = load_submission(args.submission)
    name = args.name or args.submission.stem

    verifier = Verifier(problem=args.problem, leaderboard_path=args.leaderboard)
    verifier.verify(model, name=name, lookback=args.lookback)


if __name__ == "__main__":
    main()
