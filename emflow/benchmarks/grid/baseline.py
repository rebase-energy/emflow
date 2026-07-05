"""Reference baseline for the grid problems: seasonal-naive.

Forecast = the target's value 24 hours before the target time, falling back
to the value one week before when the daily lag is not yet knowable at the
origin (the last two horizons of each origin), and to the trailing 7-day mean
when both lags are missing. Uses only the target's own past, so it works
unchanged on every ``grid:<variable>-<zone>`` variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from emflow.features import Lag, Rolling
from emflow.models.predictor import FeaturePredictor


class SeasonalNaive(FeaturePredictor):
    def __init__(self, field=None, name="seasonal-naive"):
        self.field = field
        super().__init__(features=self._specs(field or "demand"), name=name)

    @staticmethod
    def _specs(field):
        return (Lag(field, ("24h", "168h")), Rolling(field, "7D", "mean"))

    def bind(self, problem):
        super().bind(problem)
        if self.field is None:
            variable = problem.name.split(":", 1)[1].partition("-")[0]
            self.field = variable
            self.features = self._specs(variable)
        return self

    def predict_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        day, week, roll = (X.iloc[:, i].to_numpy(dtype=float) for i in range(3))
        vals = np.where(np.isfinite(day), day,
                        np.where(np.isfinite(week), week, roll))
        return pd.DataFrame({"point": vals}, index=X.index)


def get_model() -> SeasonalNaive:
    return SeasonalNaive()
