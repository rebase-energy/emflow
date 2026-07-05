"""Reference baseline for HEFTCom2024: capacity-normalized binned quantiles.

Hornsea-1's available capacity moved a lot over the data span (including a
~50% REMIT outage across the competition period), so any model of *absolute*
generation trained on history over-predicts. This baseline therefore predicts
capacity **ratios** and rescales by a trailing-max proxy of currently available
capacity — both availability-correct by construction:

* wind: empirical quantiles of ``wind_credit / trailing-14d-max(wind_credit)``
  per forecast-wind-speed decile, rescaled by the trailing max at the origin;
* solar: the same with radiation bands and the solar trailing max;
* total: comonotonic sum of the two quantile sets (crude, standard baseline).

Numpy-only. Agents are meant to beat this — and then the historical field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from emflow.data import DataFeed
from emflow.features import ForecastField, Rolling
from emflow.features.materialize import supervised_frame
from emflow.models.predictor import FeaturePredictor

from .problem import QUANTILES, TARGET_FIELD

WIND_SPEED = "hornsea_nwp_wind_speed_100"
RADIATION = "pes10_nwp_solar_down_radiation"
WIND_CAP = "energy_roll_14D_max"          # trailing max of wind credit (capacity proxy)
SOLAR_CAP = "energy_roll_14D_max_solar"   # named via dedicated spec below


class _RollingNamed(Rolling):
    """Rolling spec with an explicit output name (two rolling features over
    different columns of the same field would otherwise collide)."""

    def __init__(self, field, window, agg, column, name):
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "agg", agg)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "_name", name)

    def names(self):
        return [self._name]


class BinnedQuantileBaseline(FeaturePredictor):
    output_kind = "quantiles"
    quantiles = QUANTILES

    features = (
        ForecastField("hornsea_nwp", ["wind_speed_100"], interp="time"),
        ForecastField("pes10_nwp", ["solar_down_radiation"], interp="time"),
        _RollingNamed("energy", "14D", "max", "wind_mwh_credit", WIND_CAP),
        _RollingNamed("energy", "14D", "max", "solar_mwh_credit", SOLAR_CAP),
    )

    def __init__(self, n_wind_bins=10, n_solar_bins=4, train_window="730D",
                 name="binned-quantiles"):
        super().__init__(name=name)
        self.n_wind_bins = n_wind_bins
        self.n_solar_bins = n_solar_bins
        self.train_window = pd.Timedelta(train_window)
        self.wind_edges = None
        self.solar_edges = None
        self.wind_table = {}    # wind-speed bin -> ratio quantiles
        self.solar_table = {}   # radiation band -> ratio quantiles
        self.wind_fallback = None
        self.solar_fallback = None

    # -- training ---------------------------------------------------------------

    def fit(self, train):
        origins = [o for o in self.problem.schedule.origins(None, train.asof)
                   if o.target_end < train.asof
                   and o.target_start >= train.asof - self.train_window]
        feed = DataFeed(self.problem.load_dataset())
        X, y_wind = supervised_frame(feed, self.features, origins,
                                     TARGET_FIELD, "wind_mwh_credit")
        _, y_solar = supervised_frame(feed, self.features, origins,
                                      TARGET_FIELD, "solar_mwh_credit")
        q = np.asarray(self.quantiles)

        # Wind: ratio to trailing capacity proxy, binned by forecast wind speed.
        wind = X[WIND_SPEED].to_numpy(float)
        cap = X[WIND_CAP].to_numpy(float)
        ratio = y_wind.to_numpy(float) / np.where(cap > 1e-9, cap, np.nan)
        ok = np.isfinite(wind) & np.isfinite(ratio)
        self.wind_edges = np.quantile(wind[ok], np.linspace(0, 1, self.n_wind_bins + 1)[1:-1])
        self.wind_fallback = np.quantile(ratio[ok], q)
        bins = np.digitize(wind[ok], self.wind_edges)
        self.wind_table = {b: np.quantile(ratio[ok][bins == b], q)
                           for b in np.unique(bins) if (bins == b).sum() >= 50}

        # Solar: ratio to trailing peak, banded by forecast radiation (band 0 = dark).
        rad = X[RADIATION].to_numpy(float)
        scap = X[SOLAR_CAP].to_numpy(float)
        sratio = y_solar.to_numpy(float) / np.where(scap > 1e-9, scap, np.nan)
        ok = np.isfinite(rad) & np.isfinite(sratio)
        positive = rad[ok][rad[ok] > 1e-9]
        self.solar_edges = (np.quantile(positive, np.linspace(0, 1, self.n_solar_bins)[1:-1])
                            if len(positive) else np.array([]))
        self.solar_fallback = np.quantile(sratio[ok], q)
        sbins = self._solar_bin(rad[ok])
        self.solar_table = {b: np.quantile(sratio[ok][sbins == b], q)
                            for b in np.unique(sbins) if (sbins == b).sum() >= 50}
        return self

    def _solar_bin(self, rad):
        bands = (np.digitize(rad, self.solar_edges) + 1 if len(self.solar_edges)
                 else np.ones(len(rad), dtype=int))
        return np.where(rad <= 1e-9, 0, bands)

    # -- prediction ---------------------------------------------------------------

    def predict_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        wind = X[WIND_SPEED].to_numpy(float)
        rad = X[RADIATION].to_numpy(float)
        wcap = np.nan_to_num(X[WIND_CAP].to_numpy(float))
        scap = np.nan_to_num(X[SOLAR_CAP].to_numpy(float))

        wind_ratio = np.tile(self.wind_fallback, (n, 1))
        ok = np.isfinite(wind)
        wbins = np.digitize(wind[ok], self.wind_edges)
        wind_ratio[np.flatnonzero(ok)] = [self.wind_table.get(b, self.wind_fallback)
                                          for b in wbins.tolist()]

        solar_ratio = np.tile(self.solar_fallback, (n, 1))
        ok = np.isfinite(rad)
        sbins = self._solar_bin(rad[ok])
        solar_ratio[np.flatnonzero(ok)] = [self.solar_table.get(b, self.solar_fallback)
                                           for b in sbins.tolist()]

        total = wind_ratio * wcap[:, None] + solar_ratio * scap[:, None]
        total = np.clip(total, 0.0, None)
        total.sort(axis=1)
        return pd.DataFrame(total, index=X.index, columns=list(self.quantiles))


class MedianBidBaseline(BinnedQuantileBaseline):
    """Trading baseline: bid the forecast median.

    The risk-neutral bid absent a view on the day-ahead/imbalance price
    spread. Beating it requires forecasting the spread — which is the
    Trading Track's actual game.
    """

    output_kind = "point"

    def __init__(self, name="median-bid", **kwargs):
        super().__init__(name=name, **kwargs)

    def predict_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        q = super().predict_tabular(X)
        return pd.DataFrame({"bid": q[0.5]}, index=X.index)


def get_model() -> BinnedQuantileBaseline:
    """Submission-convention factory: a fresh, untrained baseline."""
    return BinnedQuantileBaseline()
