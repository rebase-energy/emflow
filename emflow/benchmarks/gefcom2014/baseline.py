"""Reference baseline for GEFCom2014: hour-of-day climatology quantiles.

For each origin, the empirical 1%..99% quantiles of the target column by hour
of day over a trailing window of available history — the classic "climatology"
probabilistic baseline. Works unchanged on all four tracks because it only
uses the target's own past; beating it requires actually using the provided
explanatory variables (temperature stations, load forecasts, ECMWF fields),
which is where the historical field earned their ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from emflow.models.predictor import Predictor

from .problem import QUANTILES, TRACKS


class ClimatologyQuantiles(Predictor):
    output_kind = "quantiles"
    quantiles = QUANTILES

    def __init__(self, window="120D", quantiles=None, target_field=None,
                 name="climatology-quantiles"):
        super().__init__(name=name)
        self.window = window
        if quantiles is not None:
            self.quantiles = tuple(quantiles)
        self.target_field = target_field

    def _target_field(self) -> str:
        if self.target_field is not None:
            return self.target_field
        track = self.problem.name.split(":")[1]
        return TRACKS[track]["target_field"]

    def predict(self, obs) -> pd.DataFrame:
        hist = obs.history(self._target_field(), window=self.window)
        series = hist[obs.column].dropna()
        q = np.asarray(self.quantiles)

        by_hour = {}
        fallback = (np.quantile(series.to_numpy(), q) if len(series)
                    else np.zeros(len(q)))
        if len(series):
            for hour, vals in series.groupby(series.index.hour):
                if len(vals) >= 10:
                    by_hour[hour] = np.quantile(vals.to_numpy(), q)

        rows = [by_hour.get(t.hour, fallback) for t in obs.target_index]
        return pd.DataFrame(np.vstack(rows), index=obs.target_index,
                            columns=list(self.quantiles))


def get_model() -> ClimatologyQuantiles:
    """Submission-convention factory: a fresh, untrained baseline."""
    return ClimatologyQuantiles()
