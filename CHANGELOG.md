# Changelog

## 0.3.x (unreleased)

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
