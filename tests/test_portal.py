"""The leak-proof guarantee: the portal never serves anything not knowable at asof."""

import numpy as np
import pandas as pd
import pytest

from emflow.data import DataPortal, Dataset, Field


def hourly_index(start, periods):
    return pd.date_range(start, periods=periods, freq="1h", tz="UTC")


@pytest.fixture
def dataset():
    idx = hourly_index("2025-01-01", 100)
    actual = pd.DataFrame({"power": np.arange(100.0)}, index=idx)

    # Two NWP runs a day at 00:00 and 12:00, each covering the next 48 hours,
    # disseminated 6 hours after issue.
    rows = []
    for issue in pd.date_range("2025-01-01", periods=6, freq="12h", tz="UTC"):
        for h in range(1, 49):
            valid = issue + pd.Timedelta(hours=h)
            rows.append((issue, valid, float(h)))
    nwp = pd.DataFrame(rows, columns=["issue", "valid", "wind_speed"])

    return Dataset(
        name="toy",
        fields={
            "power": Field("power", actual, availability_lag="1h"),
            "nwp": Field.forecast("nwp", nwp, issue_col="issue", valid_col="valid",
                                  availability_lag="6h"),
            "meta": Field.static("meta", pd.DataFrame({"capacity": [10.0]})),
        },
    )


class TestHistory:
    def test_availability_lag_hides_recent_values(self, dataset):
        portal = DataPortal(dataset)
        asof = pd.Timestamp("2025-01-02 00:00", tz="UTC")
        hist = portal.history(asof, "power")
        # lag 1h: value stamped 23:00 is knowable at 00:00, value at 00:00 is not
        assert hist.index.max() == pd.Timestamp("2025-01-01 23:00", tz="UTC")

    def test_nothing_after_asof_property(self, dataset):
        portal = DataPortal(dataset)
        for asof in hourly_index("2025-01-01", 100)[::7]:
            hist = portal.history(asof, "power")
            assert (hist.index + dataset.field("power").availability_lag <= asof).all()

    def test_window_is_efficiency_only(self, dataset):
        portal = DataPortal(dataset)
        asof = pd.Timestamp("2025-01-03 00:00", tz="UTC")
        full = portal.history(asof, "power")
        windowed = portal.history(asof, "power", window="24h")
        assert windowed.index.max() == full.index.max()  # same cutoff
        assert len(windowed) <= 25

    def test_kind_mismatch_raises(self, dataset):
        portal = DataPortal(dataset)
        with pytest.raises(ValueError, match="not an actual field"):
            portal.history("2025-01-02", "nwp")


class TestForecasts:
    def test_only_disseminated_runs_served(self, dataset):
        portal = DataPortal(dataset)
        # At 05:00 on Jan 1 the 00:00 run (available 06:00) is NOT out yet.
        fc = portal.forecasts(pd.Timestamp("2025-01-01 05:00", tz="UTC"), "nwp")
        assert fc.empty
        # At 06:00 it is.
        fc = portal.forecasts(pd.Timestamp("2025-01-01 06:00", tz="UTC"), "nwp")
        assert not fc.empty
        assert (fc["issue_time"] == pd.Timestamp("2025-01-01 00:00", tz="UTC")).all()

    def test_latest_run_wins_per_valid_time(self, dataset):
        portal = DataPortal(dataset)
        asof = pd.Timestamp("2025-01-02 00:00", tz="UTC")  # 00:00 + 12:00 Jan1 runs out
        fc = portal.forecasts(asof, "nwp")
        overlap = pd.Timestamp("2025-01-02 06:00", tz="UTC")  # covered by both runs
        row = fc.loc[overlap]
        assert row["issue_time"] == pd.Timestamp("2025-01-01 12:00", tz="UTC")
        # horizon from the later run: 18h, not 30h
        assert row["wind_speed"] == 18.0

    def test_never_serves_future_issues(self, dataset):
        portal = DataPortal(dataset)
        lag = dataset.field("nwp").availability_lag
        for asof in pd.date_range("2025-01-01", periods=20, freq="7h", tz="UTC"):
            fc = portal.forecasts(asof, "nwp")
            if not fc.empty:
                assert (fc["issue_time"] + lag <= asof).all()


class TestSettlement:
    def test_actuals_between_respects_availability(self, dataset):
        portal = DataPortal(dataset)
        start, end = "2025-01-01 20:00", "2025-01-02 04:00"
        asof = pd.Timestamp("2025-01-02 00:00", tz="UTC")
        got = portal.actuals_between("power", start, end, asof=asof)
        assert got.index.max() == pd.Timestamp("2025-01-01 23:00", tz="UTC")
        # Later, the full window settles.
        later = portal.actuals_between("power", start, end,
                                       asof=pd.Timestamp("2025-01-02 06:00", tz="UTC"))
        assert later.index.max() == pd.Timestamp("2025-01-02 04:00", tz="UTC")


class TestTimeView:
    def test_view_is_frozen_and_consistent(self, dataset):
        portal = DataPortal(dataset)
        asof = pd.Timestamp("2025-01-02 00:00", tz="UTC")
        view = portal.view(asof)
        pd.testing.assert_frame_equal(view.history("power"), portal.history(asof, "power"))
        pd.testing.assert_frame_equal(view.forecasts("nwp"), portal.forecasts(asof, "nwp"))
        assert view.static("meta")["capacity"].iloc[0] == 10.0


class TestField:
    def test_series_coerced_and_sorted(self):
        s = pd.Series([2.0, 1.0], index=hourly_index("2025-01-01", 2)[::-1])
        f = Field("x", s)
        assert isinstance(f.frame, pd.DataFrame)
        assert f.frame.index.is_monotonic_increasing

    def test_bad_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            Field("x", pd.DataFrame(index=hourly_index("2025-01-01", 1)), kind="nope")

    def test_forecast_requires_multiindex(self):
        with pytest.raises(TypeError, match="MultiIndex"):
            Field("x", pd.DataFrame({"a": [1.0]}, index=hourly_index("2025-01-01", 1)),
                  kind="forecast")

    def test_dataset_coerces_raw_frames(self):
        idx = hourly_index("2025-01-01", 3)
        ds = Dataset(name="d", fields={"y": pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=idx)})
        assert ds.field("y").kind == "actual"
