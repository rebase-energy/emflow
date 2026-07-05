"""Experiment: the one evaluation loop (backtrader's Cerebro).

Both trust levels share this orchestrator — the Verifier is this plus
submission policy. Two execution modes over identical settlement logic:

``event``
    reset → per origin: observe → predict → step → settle → analyze. Always
    correct; required for online-learning models.
``vectorized``
    For models with ``supports_batch`` (:class:`FeaturePredictor`): the
    feature matrix for *all* origins is materialized in one shot, the model
    predicts once as a batch, and the precomputed actions are replayed through
    the same environment. Orders of magnitude fewer model calls; settlement,
    validation and rewards are byte-identical because the env does them in
    both modes.

``mode="auto"`` picks vectorized when the model supports it.
"""

from __future__ import annotations

import typing as t

import pandas as pd

from .analyzers import Analyzer, default_analyzers
from .result import Result


class Experiment:
    def __init__(self, problem, model, analyzers: t.Union[str, t.Sequence[Analyzer], None] = "default",
                 split: str = "validation"):
        self.problem = problem
        self.model = model
        self.split = split
        if analyzers == "default":
            self.analyzers = default_analyzers(model)
        else:
            self.analyzers = list(analyzers or [])

    def run(self, mode: str = "auto") -> Result:
        if mode == "auto":
            mode = "vectorized" if getattr(self.model, "supports_batch", False) else "event"
        if mode not in ("event", "vectorized"):
            raise ValueError(f"mode must be 'auto', 'event' or 'vectorized', got {mode!r}")

        env = self.problem.env(self.split)
        obs, info = env.reset()

        for analyzer in self.analyzers:
            analyzer.setup(env.feed, env.target_field, env.target_column, env.objective)

        if hasattr(self.model, "bind"):
            self.model.bind(self.problem)
        if "train" in info:
            self.model.fit(info["train"])

        if mode == "vectorized":
            actions = self._batch_actions(env)
        else:
            actions = None

        records = []
        step = 0
        while True:
            action = actions[step] if actions is not None else self.model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            for record in info["settled"]:
                records.append(record)
                for analyzer in self.analyzers:
                    analyzer.on_settlement(record)
            step += 1
            if terminated or truncated:
                break

        return self._build_result(env, records, mode)

    # -- vectorized fast path -------------------------------------------------------

    def _batch_actions(self, env) -> t.List[pd.DataFrame]:
        from ..features.materialize import materialize

        if not getattr(self.model, "supports_batch", False):
            raise ValueError(
                f"model {self.model.name!r} does not support batch prediction; "
                f"use mode='event'"
            )
        X = materialize(env.feed, self.model.features, env.origins)
        out = self.model.predict_tabular(X)
        if not isinstance(out.index, pd.MultiIndex):
            raise TypeError(
                "predict_tabular must preserve the (asof, target_time) index "
                "for batch execution"
            )
        by_asof = dict(tuple(out.groupby(level="asof", sort=False)))
        return [by_asof[o.asof].droplevel("asof") for o in env.origins]

    # -- assembly ---------------------------------------------------------------------

    def _build_result(self, env, records, mode: str) -> Result:
        preds, rows = [], []
        for r in records:
            block = r.prediction.copy()
            block.index = pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([r.origin.asof] * len(block)), block.index],
                names=("asof", "target_time"),
            )
            preds.append(block)
            rows.append({
                "asof": r.origin.asof,
                "target_start": r.origin.target_start,
                "target_end": r.origin.target_end,
                "n_scored": int(r.actuals.notna().sum()),
                "score": r.score,
            })

        settlements = pd.DataFrame(rows).set_index("asof") if rows else pd.DataFrame(
            columns=["target_start", "target_end", "n_scored", "score"])
        analysis = {a.name: out for a in self.analyzers if (out := a.finalize())}

        return Result(
            problem=self.problem.name,
            model=getattr(self.model, "name", type(self.model).__name__),
            objective=env.objective.name,
            lower_is_better=env.objective.lower_is_better,
            split=self.split,
            mode=mode,
            predictions=pd.concat(preds) if preds else pd.DataFrame(),
            settlements=settlements,
            analysis=analysis,
            aggregate=getattr(env.objective.metric, "aggregate", "mean"),
        )
