"""Leak-proof verification of submitted forecasting models.

A :class:`Verifier` evaluates a user-submitted :class:`emflow.Predictor` against an
emflow problem (default ``swedish-temperatures:ar``) and reports how well it performs
— with **no possibility of data leakage**:

* The submitter sends a *fresh, untrained* model (via ``get_model()``); the verifier
  trains it itself on the official training split, so the model can never be fit on
  test data.
* Scoring uses **strict walk-forward**: to forecast hour ``t`` the model is handed only
  the series strictly before ``t`` (the value at ``t`` is hidden), so it is physically
  impossible to peek at the present or future.

Results are printed as a scorecard and appended to a leaderboard CSV so multiple
submissions can be compared.
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

import emflow as ef
from emflow.problems.objective import MeanSquaredError

DEFAULT_LEADERBOARD = Path(__file__).resolve().parents[2] / "submissions" / "leaderboard.csv"
MODE = "strict walk-forward (leakage impossible by construction)"


def _forecast_at(out, t):
    """Extract the scalar 1-step-ahead forecast for timestamp ``t`` from a model's
    ``predict`` output, with clear errors on interface violations."""
    if isinstance(out, pd.DataFrame):
        if t not in out.index:
            raise ValueError(f"predict() output has no row for the forecast time {t}")
        row = out.loc[t]
        return float(row.iloc[0])
    if isinstance(out, pd.Series):
        if t not in out.index:
            raise ValueError(f"predict() output has no entry for the forecast time {t}")
        return float(out.loc[t])
    raise TypeError("predict() must return a pandas DataFrame or Series indexed by time")


class Verifier:
    """Train and strictly evaluate a submitted predictor on an emflow problem."""

    def __init__(self, problem="swedish-temperatures:ar", leaderboard_path=DEFAULT_LEADERBOARD):
        self.problem_name = problem
        self.dataset, self.env, self.objective = ef.load_problem(problem)
        self.leaderboard_path = Path(leaderboard_path) if leaderboard_path else None

    # -- scoring ------------------------------------------------------------
    def verify(self, model, name=None, lookback=None, record=True, verbose=True) -> dict:
        """Evaluate ``model`` and return a result dict.

        Parameters
        ----------
        model : emflow.Predictor
            A *fresh, untrained* predictor. It is trained here on the official split.
        name : str, optional
            Submission name (defaults to ``model.name`` or the class name).
        lookback : int, optional
            If set, only the trailing ``lookback`` hours of history are passed at each
            step (speeds up slow models). Default ``None`` = full history (safe for any
            model). Does not affect leakage — only how much *past* is provided.
        record : bool
            Append the result to the leaderboard CSV.
        """
        if not isinstance(model, ef.Predictor):
            raise TypeError(
                f"submission must be an emflow.Predictor subclass, got {type(model).__name__}"
            )
        if not callable(getattr(model, "predict", None)):
            raise TypeError("submission must implement predict(input)")
        name = name or getattr(model, "name", None) or type(model).__name__

        # (b) Controlled split straight from the environment.
        init, _ = self.env.reset()
        train = init["target"]
        _, test_target, _ = self.env.step()
        full = pd.concat([train, test_target]).sort_index()
        col = full.columns[0]
        series = full[col]

        # (c) Train on the training split ONLY.
        model.train(train)

        # (d) Strict walk-forward: forecast each test hour from the past only.
        test_t = test_target[col].dropna().index
        yhat = pd.Series(index=test_t, dtype=float)
        for t in test_t:
            hist = full.loc[:t].copy()        # nothing after t exists in the slice
            if lookback:
                hist = hist.iloc[-(lookback + 1):]
            hist.loc[t] = np.nan              # hide the answer at t
            yhat.loc[t] = _forecast_at(model.predict(hist), t)

        y = test_target[col].reindex(test_t)

        # (e) Metrics — the problem's objective (primary) + RMSE, plus baselines.
        valid = int((y.notna() & yhat.notna()).sum())
        if valid == 0:
            raise ValueError("model produced no valid forecasts on the test set")
        mse = MeanSquaredError()
        mae = float(self.objective.calculate(y, yhat))
        rmse = float(mse.calculate(y, yhat, squared=False))
        baselines = {
            "Persistence (t-1)": series.shift(1).reindex(test_t),
            "Seasonal naive (t-24)": series.shift(24).reindex(test_t),
        }
        base_scores = {
            n: (float(self.objective.calculate(y, p)), float(mse.calculate(y, p, squared=False)))
            for n, p in baselines.items()
        }
        persistence_mae = base_scores["Persistence (t-1)"][0]
        beats = mae < persistence_mae

        result = {
            "submission": name,
            "problem": self.problem_name,
            "mode": MODE,
            "metric_name": self.objective.name,
            "mae": mae,
            "rmse": rmse,
            "n_points": valid,
            "persistence_mae": persistence_mae,
            "beats_persistence": bool(beats),
            "baselines": base_scores,
        }

        if verbose:
            self._print_scorecard(result)
        if record and self.leaderboard_path:
            self._append_leaderboard(result)
        return result

    # -- reporting ----------------------------------------------------------
    def _print_scorecard(self, r):
        print(f"\n{'='*60}")
        print(f"Submission : {r['submission']}")
        print(f"Problem    : {r['problem']}")
        print(f"Mode       : {r['mode']}")
        print(f"Scored on  : {r['n_points']:,} test hours")
        print(f"{'-'*60}")
        print(f"{'Model':<28}{'MAE (°C)':>12}{'RMSE (°C)':>12}")
        print(f"{'-'*60}")
        print(f"{r['submission']:<28}{r['mae']:>12.3f}{r['rmse']:>12.3f}")
        for n, (m, rm) in r["baselines"].items():
            print(f"{n:<28}{m:>12.3f}{rm:>12.3f}")
        print(f"{'-'*60}")
        verdict = "PASS — beats persistence" if r["beats_persistence"] else "FAIL — does not beat persistence"
        print(f"Verdict    : {verdict}")
        print(f"Leakage    : impossible by construction (strict walk-forward)")
        print(f"{'='*60}\n")

    def _append_leaderboard(self, r):
        path = self.leaderboard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["submitted_at", "submission", "problem", "mode", "metric_name",
                  "mae", "rmse", "n_points", "persistence_mae", "beats_persistence"]
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({
                "submitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "submission": r["submission"], "problem": r["problem"], "mode": r["mode"],
                "metric_name": r["metric_name"], "mae": round(r["mae"], 4),
                "rmse": round(r["rmse"], 4), "n_points": r["n_points"],
                "persistence_mae": round(r["persistence_mae"], 4),
                "beats_persistence": r["beats_persistence"],
            })
        print(f"Appended to leaderboard: {path}")
