"""TradingEnv: day-ahead bidding on top of the rolling-origin loop.

The action gains a ``"bid"`` column (MWh per settlement period, committed to
the day-ahead market at the origin). Settlement follows HEFTCom24's rules:
each period earns

    bid × DA_price + (actual − bid) × SS_price − 0.07·(actual − bid)²

so imbalances settle at the system price plus a quadratic penalty (verified
against the official per-period revenues in the competition archive). Reward
is the (positive) revenue, credited only once both the generation actuals
*and* the prices are knowable — you find out how a trade went when the
settlement data lands, exactly as in reality.
"""

from __future__ import annotations

import typing as t

import pandas as pd

from ..problems.metrics import TradingRevenue
from .forecast import ForecastEnv, Origin

BID_COL = "bid"


class TradingEnv(ForecastEnv):
    """ForecastEnv + market settlement.

    Extra parameters: ``prices_field`` (an actual field with day-ahead and
    system-price columns) and the two column names.
    """

    def __init__(self, *args, prices_field: str = "prices",
                 da_column: str = "da_price", ss_column: str = "ss_price",
                 **kwargs):
        kwargs.setdefault("quantiles", None)
        super().__init__(*args, **kwargs)
        self.prices_field = prices_field
        self.da_column = da_column
        self.ss_column = ss_column
        prices_lag = self.feed.dataset.field(prices_field).settlement_lag
        self._settle_lag = max(self._settle_lag, prices_lag)

    def _validate(self, action, origin: Origin) -> pd.DataFrame:
        if not isinstance(action, pd.DataFrame):
            raise TypeError("action must be a DataFrame indexed by target time")
        if BID_COL not in action.columns:
            raise ValueError(f"trading action must have a {BID_COL!r} column, "
                             f"got {list(action.columns)}")
        missing = origin.target_index.difference(action.index)
        if len(missing):
            raise ValueError(f"action is missing bids for {len(missing)} target times")
        return action.reindex(origin.target_index)

    def _score(self, origin, prediction, actuals, asof) -> float:
        prices = self.feed.actuals_between(
            self.prices_field, origin.target_start, origin.target_end, asof=asof,
        ).reindex(origin.target_index)
        metric = self.objective.metric
        if not isinstance(metric, TradingRevenue):
            raise TypeError("TradingEnv requires a TradingRevenue objective")
        revenue = metric.revenue(
            actuals, prediction[BID_COL],
            prices[self.da_column], prices[self.ss_column],
        )
        return float(revenue.sum())
