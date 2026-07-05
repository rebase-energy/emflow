"""Verify a submitted forecasting model and score it on an emflow problem.

Usage:
    # local file
    python scripts/verify_submission.py <path/to/submission.py> [...]

    # straight from HuggingFace (intern uploads submission.py to a HF repo)
    python scripts/verify_submission.py hf://<owner>/<repo>/submission.py \
        [--revision COMMIT_OR_TAG] [--repo-type model|dataset] [...]

    # ...also: [--problem swedish-temperatures:ar] [--name NAME]
    #          [--leaderboard submissions/leaderboard.csv] [--lookback HOURS]

The submission file must expose ``get_model() -> emflow.Predictor`` returning a
*fresh, untrained* predictor (or a module-level ``model = ...`` instance). The
verifier fits it on the official training view and scores it on the holdout
split, where every observation is served through the point-in-time DataFeed —
so the reported number cannot be inflated by data leakage.

SECURITY: this imports and executes the submission file (local or downloaded). Only
run code from people you trust, and prefer pinning ``--revision`` to a commit SHA.
"""

import argparse
import os
from pathlib import Path

from emflow.run.submission import load_submission
from emflow.run.verifier import Verifier, DEFAULT_LEADERBOARD


def resolve_to_local(ref: str, revision=None, repo_type="model") -> Path:
    """Resolve a submission reference to a local file path.

    Accepts a plain local path, or a ``hf://<owner>/<repo>/<path-in-repo>`` URI which is
    downloaded from the HuggingFace Hub (uses HF_TOKEN/HUGGINGFACE_TOKEN if set, for
    private repos).
    """
    if not ref.startswith("hf://"):
        return Path(ref)

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as e:
        raise SystemExit(
            "huggingface_hub is required for hf:// submissions — "
            "install it with:  uv sync --extra submissions"
        ) from e

    parts = ref[len("hf://"):].split("/")
    if len(parts) < 2:
        raise SystemExit("hf:// reference must be hf://<owner>/<repo>[/<file>]")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) or "submission.py"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    print(f"Downloading {filename} from HF {repo_type} repo {repo_id}"
          f"{' @ ' + revision if revision else ''} ...")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                repo_type=repo_type, revision=revision, token=token))


def main():
    ap = argparse.ArgumentParser(description="Verify a submitted Predictor model.")
    ap.add_argument("submission", help="local path OR hf://<owner>/<repo>/submission.py")
    ap.add_argument("--problem", default="swedish-temperatures:ar")
    ap.add_argument("--name", default=None, help="submission name for the scorecard/leaderboard")
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    ap.add_argument("--split", default="holdout", choices=["holdout", "validation"],
                    help="which split to score (default: holdout — the official one)")
    ap.add_argument("--revision", default=None, help="HF commit SHA / tag / branch to pin")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"],
                    help="HF repo type for hf:// references (default: model)")
    args = ap.parse_args()

    path = resolve_to_local(args.submission, revision=args.revision, repo_type=args.repo_type)
    print(f"⚠  Importing and running {path} — only do this for trusted code.")
    model = load_submission(path)
    name = args.name or path.stem

    verifier = Verifier(problem=args.problem, leaderboard_path=args.leaderboard,
                        split=args.split)
    verifier.verify(model, name=name)


if __name__ == "__main__":
    main()
