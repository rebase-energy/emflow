"""Train and evaluate a least-squares AR temperature forecaster (Stockholm).

Loads the emflow problem ``swedish-temperatures:ar``, runs the built-in AR
baseline through an :class:`emflow.Experiment` (vectorized mode), and prints
the scorecard: MAE on the validation split plus persistence skill.

Run:
    python scripts/ar_forecast_swedish_temperatures.py [--split validation|holdout]
"""

from __future__ import annotations

import argparse

import emflow as ef
from emflow.examples.swedish_temperatures.predictor import ARPredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation", "holdout"])
    ap.add_argument("--mode", default="auto", choices=["auto", "event", "vectorized"])
    args = ap.parse_args()

    problem = ef.load_problem("swedish-temperatures:ar")
    dataset = problem.load_dataset()
    model = ARPredictor()

    start, end = problem.splits.period(args.split)
    print(f"Station    : {dataset.collection.members[0].name}")
    print(f"Train      : < {problem.splits.train_end}")
    print(f"Evaluate   : {start} -> {end}  [{args.split}]")
    print(f"Lags       : {model.lags}")

    result = ef.Experiment(problem, model).run(mode=args.mode)

    stats = result.stats()
    print(f"\n{result!r}")
    print(f"Mode       : {result.mode}")
    print(f"Scored     : {result.n_scored:,} hours over {result.n_origins:,} origins")
    print(f"MAE        : {result.score:.4f} °C")
    skill = result.analysis.get("PersistenceSkill", {})
    if skill:
        print(f"Persistence: {skill['persistence_score']:.4f} °C  "
              f"(skill {skill.get('skill', float('nan')):+.2%} — "
              f"{'beats' if skill['beats_persistence'] else 'does NOT beat'} baseline)")


if __name__ == "__main__":
    main()
