"""Grid loader and problem family — all offline (pre-seeded cache, no API)."""

import numpy as np
import pandas as pd
import pytest

from emflow.data.io import grid
from emflow.benchmarks.grid import problem as grid_problem


def _seed_cache(monkeypatch, tmp_path, zone, variable, start, end):
    monkeypatch.setenv("EMFLOW_GRID_CACHE", str(tmp_path))
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    index = pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC")
    frame = pd.DataFrame(
        {variable: 100 + 10 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)},
        index=pd.DatetimeIndex(index, name="datetime"),
    )
    path = (tmp_path /
            f"{zone}_{variable}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_1H.parquet")
    frame.to_parquet(path)
    return frame


class TestLoadSeries:
    def test_cache_hit_needs_no_network_or_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EMFLOW_GRID_API_KEY", raising=False)
        monkeypatch.delenv("GRID_API_KEY", raising=False)
        monkeypatch.setenv("EMFLOW_GRID_OFFLINE", "1")
        seeded = _seed_cache(monkeypatch, tmp_path, "DE-LU", "demand",
                             "2026-01-01", "2026-02-01")
        out = grid.load_series("DE-LU", "demand", "2026-01-01", "2026-02-01")
        pd.testing.assert_frame_equal(out, seeded, check_freq=False)

    def test_offline_cache_miss_is_a_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EMFLOW_GRID_CACHE", str(tmp_path))
        monkeypatch.setenv("EMFLOW_GRID_OFFLINE", "1")
        with pytest.raises(RuntimeError, match="EMFLOW_GRID_OFFLINE"):
            grid.load_series("FR", "demand", "2026-01-01", "2026-02-01")

    def test_fetch_without_key_is_a_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EMFLOW_GRID_CACHE", str(tmp_path))
        monkeypatch.delenv("EMFLOW_GRID_OFFLINE", raising=False)
        monkeypatch.delenv("EMFLOW_GRID_API_KEY", raising=False)
        monkeypatch.delenv("GRID_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key"):
            grid.fetch_series("FR", "demand", "2026-01-01", "2026-02-01")

    def test_unknown_variable_rejected(self):
        with pytest.raises(KeyError, match="unknown variable"):
            grid.fetch_series("DE-LU", "price", "2026-01-01", "2026-02-01")


class TestGridProblems:
    def test_variants_registered(self):
        import emflow as ef
        from emflow.benchmarks.grid.zones import ZONE_REGISTRY

        names = [n for n in ef.list_problems() if n.startswith("grid:")]
        expected = sorted(
            f"grid:{v}-{slug}"
            for slug, entry in ZONE_REGISTRY.items() for v in entry["variables"]
        )
        assert names == expected
        assert "grid:demand-de-lu" in names  # pilot zones must stay qualified
        assert "grid:solar-se-se4" in names

    def test_parse_variant(self):
        assert grid_problem.parse_variant("demand-de-lu") == ("demand", "DE-LU")
        assert grid_problem.parse_variant("wind-se-se4") == ("wind", "SE-SE4")
        with pytest.raises(KeyError):
            grid_problem.parse_variant("price-de-lu")

    def test_end_to_end_eval_from_cache(self, monkeypatch, tmp_path):
        import emflow as ef
        from emflow.benchmarks.grid.baseline import get_model
        from emflow.run.experiment import Experiment

        monkeypatch.setenv("EMFLOW_GRID_OFFLINE", "1")
        _seed_cache(monkeypatch, tmp_path, "DE-LU", "demand",
                    grid_problem.DATA_END - grid_problem.HISTORY,
                    grid_problem.DATA_END)
        problem = ef.load_problem("grid:demand-de-lu")
        result = Experiment(problem, get_model(), split="validation").run()
        # the seeded series is a pure 24h sine, so seasonal-naive is exact
        assert result.score == pytest.approx(0.0, abs=1e-9)

    def test_insufficient_history_raises_not_ingested(self, monkeypatch, tmp_path):
        import emflow as ef
        from emflow.problems.registry import ProblemNotIngestedError

        monkeypatch.setenv("EMFLOW_GRID_OFFLINE", "1")
        _seed_cache(monkeypatch, tmp_path, "DE-LU", "demand",
                    grid_problem.SPLITS.train_end - pd.Timedelta(days=10),
                    grid_problem.DATA_END)
        # cache file must span the problem's full window to be picked up
        short = _seed_cache(monkeypatch, tmp_path, "DE-LU", "demand",
                            grid_problem.DATA_END - grid_problem.HISTORY,
                            grid_problem.DATA_END)
        cutoff = grid_problem.SPLITS.train_end - pd.Timedelta(days=10)
        short.loc[short.index < cutoff, "demand"] = np.nan
        path = (tmp_path / f"DE-LU_demand_"
                f"{grid_problem.DATA_END - grid_problem.HISTORY:%Y%m%dT%H%M}_"
                f"{grid_problem.DATA_END:%Y%m%dT%H%M}_1H.parquet")
        short.to_parquet(path)
        with pytest.raises(ProblemNotIngestedError, match="not enough history"):
            ef.load_problem("grid:demand-de-lu").load_dataset()
