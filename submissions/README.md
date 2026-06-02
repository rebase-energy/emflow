# Model submissions

Submit a forecasting model as **code**, and the verifier scores it on the
`swedish-temperatures:ar` problem with **no possibility of data leakage**.

## How it works

- You send a `submission.py` that exposes **`get_model() -> emflow.Predictor`** returning
  a *fresh, untrained* model.
- The verifier **trains your model itself** on the official training split (everything
  before 2026), then scores it on the held-out 2026 test set using **strict
  walk-forward**: to forecast hour `t`, your model is given only the series strictly
  before `t` (the value at `t` is hidden). You never see the test targets, and you
  cannot train on them — so the reported score is trustworthy.

## Your model contract

Subclass `emflow.Predictor` (see `template_submission.py`):

- `train(self, train_df)` — fit on the wide temperature DataFrame you're given
  (hourly UTC index, one column per station). Use only what's passed.
- `predict(self, input_df)` — return a DataFrame/Series whose entry at the **last
  timestamp** of `input_df` is your 1-step-ahead forecast for that hour. Use only past
  lags; the value at the last timestamp is deliberately `NaN`.

## Submit + self-check

1. Copy `template_submission.py`, implement `MyModel` / `get_model()`.
2. Self-check it runs and scores:

   ```bash
   python scripts/verify_submission.py submissions/template_submission.py
   ```

   This prints a scorecard (your MAE/RMSE vs persistence and seasonal-naive baselines,
   `PASS` if you beat persistence) and appends a row to `submissions/leaderboard.csv`.

3. Submit your model — either send the `submission.py` file directly, or upload it to a
   HuggingFace repo (see below).

## Submit via HuggingFace

The intern uploads `submission.py` to a HuggingFace repo and shares the repo id:

```python
from huggingface_hub import HfApi
api = HfApi()                                  # set HF_TOKEN for a private repo
api.create_repo("your-username/emflow-submission", repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="submission.py", path_in_repo="submission.py",
                repo_id="your-username/emflow-submission", repo_type="model")
```

The evaluator then verifies it straight from the Hub (needs the `submissions` extra:
`uv sync --extra submissions`):

```bash
python scripts/verify_submission.py hf://your-username/emflow-submission/submission.py \
    --revision <commit-sha>          # pin a commit for reproducibility (recommended)
# add --repo-type dataset if it was uploaded to a dataset repo
```

For a private repo, set `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) in the environment before
running. The downloaded file is then trained + scored by the same strict walk-forward
verifier, so the leakage guarantees are identical to the local path.

> **Note (for whoever runs the verifier):** `verify_submission.py` imports and executes
> the submission file — local *or downloaded from HuggingFace*. Only run code from people
> you trust, and prefer pinning `--revision` to a commit SHA.
