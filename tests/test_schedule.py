import pandas as pd
import pytest

from emflow.problems.schedule import IssueSchedule, Origin


class TestHourly:
    def test_one_step_ahead(self):
        sched = IssueSchedule.hourly(horizon="1h")
        origins = sched.origins("2026-01-01 00:00", "2026-01-01 05:00")
        # every target hour in the half-open period is covered exactly once
        targets = [o.target_index[0] for o in origins]
        assert targets == list(pd.date_range("2026-01-01 00:00", "2026-01-01 04:00", freq="1h"))
        for o in origins:
            assert o.target_index[0] == o.asof + pd.Timedelta("1h")
            assert len(o.target_index) == 1

    def test_tz_aware_period(self):
        sched = IssueSchedule.hourly()
        origins = sched.origins(pd.Timestamp("2026-01-01", tz="UTC"),
                                pd.Timestamp("2026-01-01 03:00", tz="UTC"))
        assert all(o.asof.tz is not None for o in origins)


class TestDaily:
    def test_day_ahead_pattern(self):
        # HEFTCom-style: issue at 09:00, cover 22:30–46:00 half-hourly.
        sched = IssueSchedule.daily(at="09:00", covers=("13h30min", "37h"),
                                    target_freq="30min")
        origins = sched.origins("2024-03-02 00:00", "2024-03-03 23:59")
        by_asof = {o.asof: o for o in origins}
        asof = pd.Timestamp("2024-03-02 09:00")
        assert asof in by_asof
        o = by_asof[asof]
        assert o.target_start == pd.Timestamp("2024-03-02 22:30")
        assert o.target_end == pd.Timestamp("2024-03-03 22:00")
        assert len(o.target_index) == 48

    def test_targets_clipped_to_period(self):
        sched = IssueSchedule.daily(at="09:00", covers=("13h30min", "37h"),
                                    target_freq="30min")
        origins = sched.origins("2024-03-02 00:00", "2024-03-02 23:30")
        for o in origins:
            assert o.target_end < pd.Timestamp("2024-03-02 23:30")

    def test_all_targets_after_asof(self):
        sched = IssueSchedule.daily(at="09:00", covers=("13h30min", "37h"),
                                    target_freq="30min")
        for o in sched.origins("2024-03-01", "2024-03-10"):
            assert (o.target_index > o.asof).all()


class TestExplicit:
    def test_single(self):
        idx = pd.date_range("2014-06-01", periods=24, freq="1h")
        sched = IssueSchedule.single("2014-05-31 12:00", idx)
        (o,) = sched.origins()
        assert o.asof == pd.Timestamp("2014-05-31 12:00")
        assert len(o.target_index) == 24

    def test_filtering(self):
        origins = [
            Origin(pd.Timestamp("2014-05-01"), pd.date_range("2014-05-02", periods=3, freq="1D")),
            Origin(pd.Timestamp("2014-06-01"), pd.date_range("2014-06-02", periods=3, freq="1D")),
        ]
        sched = IssueSchedule.explicit(origins)
        assert len(sched.origins("2014-06-01", "2014-07-01")) == 1


class TestValidation:
    def test_zero_horizon_rejected(self):
        with pytest.raises(ValueError, match="cover_start"):
            IssueSchedule(origin_freq="1h", cover_start="0h", cover_end="1h",
                          target_freq="1h")
