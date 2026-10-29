# Changelog

## Unreleased

- **Dependencies**: require `energydatamodel>=0.12.0,<0.13.0` (was
  `>=0.7.0,<0.8.0`), which brings `timedatamodel` 0.12.0. No code changes
  were needed: every `energydatamodel` name emflow re-exports from
  `emflow.assets` still exists in 0.12. Side effect: energydatamodel 0.12
  no longer pulls in geopandas or pvlib (pvlib is now its optional `[solar]`
  extra), so the lockfile drops geopandas, pvlib, scipy, pyproj, pyogrio,
  h5py and pytz.

## 0.3.0 (2026-07-07)

- **Integration contract**: `emflow.run.submission` is the stable surface for
  external drivers (rebase-hillclimb, rebase-toolkit): `load_submission(path)
  -> Predictor` (submission-module convention: `get_model()` factory, or a
  module-level `model`/sole Predictor instance) and
  `evaluate(problem_or_name, model_or_path, split="validation") -> Result`.
  Together with `Predictor`/`FeaturePredictor` and `Verifier` (including its
  `metadata` seam), treat signature changes to these as breaking; downstream
  packages pin `emflow>=0.3,<0.4`.
- `scripts/verify_submission.py` now imports `load_submission` from the
  package instead of defining its own copy.
- **Live grid problems** (`grid:<variable>-<zone>`): autoregressive 24h-ahead
  forecasting of demand/wind/solar per European bidding zone, backed by the
  rebase-grid API instead of HuggingFace snapshots. Cache-first loader
  (`emflow.data.io.grid`; `EMFLOW_GRID_API_KEY`, `EMFLOW_GRID_OFFLINE`,
  `EMFLOW_GRID_DATA_END`), probe-generated zone registry
  (`scripts/probe_grid_zones.py` — 43 zones / 97 variants), seasonal-naive
  reference baseline.
- **Transfer evaluation**: `emflow.sweep(solutions, problems)` scores
  submission modules across a problem family; `emflow.best_per_problem`
  picks each problem's winner — coverage from a handful of searched anchors.
