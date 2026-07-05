# Model submissions

Submit a forecasting model as **code**, and the verifier scores it on an emflow
problem (default `swedish-temperatures:ar`) with **no possibility of data
leakage**.

## How it works

- You send a `submission.py` exposing **`get_model() -> emflow.Predictor`**
  (or a module-level `model = ...` instance) returning a *fresh, untrained* model.
- The verifier **fits your model itself** on the official training view, then
  scores it origin-by-origin on the **holdout split**. Every observation your
  model sees is served through emflow's point-in-time `DataFeed`: at each
  forecast origin it contains only what was knowable at that moment — the
  targets are structurally absent, so the reported score is trustworthy.

## Your model contract

Subclass `emflow.Predictor` (see `template_submission.py` and
`example_submission.py`):

- `fit(self, train)` — `train` is a **TimeView** frozen at the training cutoff.
  Pull the training series with `train.history("temperature")`. Use only what
  it serves.
- `predict(self, obs)` — `obs` is an **Observation**: `obs.target_index` is
  what you must forecast, `obs.history("temperature")` is everything knowable
  at the origin. Return a DataFrame indexed by `obs.target_index` with a
  `"point"` column.

For faster evaluation (and richer features), subclass
`emflow.FeaturePredictor` instead: declare declarative feature specs
(`Lag`, `Rolling`, `ForecastField`, `Calendar`) and implement
`predict_tabular(X)` — the verifier then runs you in vectorized mode
(one batched call instead of thousands). See
`emflow/examples/swedish_temperatures/predictor.py` for a full example.

## Submit + self-check

1. Copy `template_submission.py`, implement `MyModel` / `get_model()`.
2. Self-check it runs and scores:

   ```bash
   python scripts/verify_submission.py submissions/template_submission.py
   ```

   This prints a scorecard (your score vs the persistence baseline, `PASS` if
   you beat it) and appends a row to `submissions/leaderboard.csv`.

3. Submit your model — send the `submission.py` directly, or upload it to a
   HuggingFace repo (below).

Iterate on the **validation** split (`--split validation`); the holdout is
scored once, at submission time. Iterating against the holdout invalidates
your score.

## Submit via HuggingFace

Upload `submission.py` to a HF repo and share the repo id:

```python
from huggingface_hub import HfApi
api = HfApi()                                  # set HF_TOKEN for a private repo
api.create_repo("your-username/emflow-submission", repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="submission.py", path_in_repo="submission.py",
                repo_id="your-username/emflow-submission", repo_type="model")
```

The evaluator verifies it straight from the Hub:

```bash
python scripts/verify_submission.py hf://your-username/emflow-submission/submission.py \
    --revision <commit-sha>          # pin a commit for reproducibility (recommended)
# add --repo-type dataset if it was uploaded to a dataset repo
```

For a private repo, set `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) before running.

> **Note (for whoever runs the verifier):** `verify_submission.py` imports and
> executes the submission file — local *or downloaded from HuggingFace*. Only
> run code from people you trust, and prefer pinning `--revision` to a commit
> SHA. For fully untrusted (agent) submissions, run in a sandbox with no
> network and keep holdout labels in a private `rb://` repo.
