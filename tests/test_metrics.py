import numpy as np
import pandas as pd
import pytest

from emflow.problems.metrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PeakTimingError,
    PinballLoss,
    RootMeanSquaredError,
)
from emflow.problems.objective import Objective


def idx(n, freq="1h", start="2024-01-01"):
    return pd.date_range(start, periods=n, freq=freq)


class TestPointMetrics:
    def test_mae(self):
        y = pd.Series([1.0, 2.0, 3.0], index=idx(3))
        p = pd.DataFrame({"point": [2.0, 2.0, 1.0]}, index=idx(3))
        assert MeanAbsoluteError().calculate(y, p) == pytest.approx(1.0)

    def test_rmse(self):
        y = pd.Series([0.0, 0.0], index=idx(2))
        p = pd.DataFrame({"point": [3.0, 4.0]}, index=idx(2))
        assert RootMeanSquaredError().calculate(y, p) == pytest.approx(np.sqrt(12.5))
        assert MeanSquaredError().calculate(y, p) == pytest.approx(12.5)

    def test_mape(self):
        y = pd.Series([100.0, 200.0], index=idx(2))
        p = pd.DataFrame({"point": [110.0, 180.0]}, index=idx(2))
        assert MeanAbsolutePercentageError().calculate(y, p) == pytest.approx(10.0)

    def test_nan_pairs_ignored(self):
        y = pd.Series([1.0, np.nan, 3.0], index=idx(3))
        p = pd.DataFrame({"point": [2.0, 5.0, np.nan]}, index=idx(3))
        assert MeanAbsoluteError().calculate(y, p) == pytest.approx(1.0)

    def test_misaligned_prediction_reindexed(self):
        y = pd.Series([1.0, 2.0], index=idx(2))
        p = pd.DataFrame({"point": [2.0, 1.0]}, index=idx(2)[::-1])
        assert MeanAbsoluteError().calculate(y, p) == pytest.approx(0.0)

    def test_median_used_as_point_for_quantile_frames(self):
        y = pd.Series([1.0], index=idx(1))
        p = pd.DataFrame({0.1: [0.0], 0.5: [2.0], 0.9: [4.0]}, index=idx(1))
        assert MeanAbsoluteError().calculate(y, p) == pytest.approx(1.0)


class TestPinball:
    def test_hand_computed(self):
        # y=10; q10 pred 12 (over: (0.1-1)*(10-12)=1.8), q90 pred 8 (under: 0.9*2=1.8)
        y = pd.Series([10.0], index=idx(1))
        p = pd.DataFrame({0.1: [12.0], 0.9: [8.0]}, index=idx(1))
        assert PinballLoss().calculate(y, p) == pytest.approx(1.8)

    def test_perfect_median_zero_loss(self):
        y = pd.Series([5.0, 7.0], index=idx(2))
        p = pd.DataFrame({0.5: [5.0, 7.0]}, index=idx(2))
        assert PinballLoss().calculate(y, p) == pytest.approx(0.0)

    def test_required_quantiles_enforced(self):
        y = pd.Series([1.0], index=idx(1))
        p = pd.DataFrame({0.5: [1.0]}, index=idx(1))
        with pytest.raises(ValueError, match="missing required quantiles"):
            PinballLoss(quantiles=[0.1, 0.5, 0.9]).calculate(y, p)

    def test_no_quantile_columns_rejected(self):
        y = pd.Series([1.0], index=idx(1))
        p = pd.DataFrame({"point": [1.0]}, index=idx(1))
        with pytest.raises(ValueError, match="no quantile columns"):
            PinballLoss().calculate(y, p)


class TestPeakTiming:
    def test_peak_hour_distance(self):
        hours = idx(48)
        y = pd.Series(0.0, index=hours)
        p = pd.DataFrame({"point": 0.0}, index=hours)
        # day 1: true peak 10:00, predicted 12:00 -> 2h; day 2: both 18:00 -> 0h
        y.iloc[10], y.iloc[24 + 18] = 1.0, 1.0
        p.iloc[12, 0], p.iloc[24 + 18, 0] = 1.0, 1.0
        assert PeakTimingError().calculate(y, p) == pytest.approx(1.0)  # mean(2, 0)


class TestObjective:
    def test_direction(self):
        lower = Objective(MeanAbsoluteError())
        assert lower.is_better(1.0, 2.0) and not lower.is_better(2.0, 1.0)
        higher = Objective(MeanAbsoluteError(), lower_is_better=False)
        assert higher.is_better(2.0, 1.0)
