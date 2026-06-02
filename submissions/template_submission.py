"""Submission template — copy this file, implement your model, and send it back.

The verifier will call ``get_model()`` to obtain a FRESH, UNTRAINED predictor, then:
  1. train it on the official training split  (model.train(train_df)),
  2. score it via strict walk-forward on the held-out 2026 test set.

You never receive the test targets, and your model is trained by the verifier on the
official split — so the score cannot be affected by data leakage.

Contract for your Predictor
---------------------------
* Subclass ``emflow.Predictor`` and set ``self.name``.
* ``train(self, train_df)``: fit on the given wide temperature DataFrame
  (index = hourly UTC timestamps, one column per station). Use only what is passed.
* ``predict(self, input_df)``: return a DataFrame (or Series) whose entry at the LAST
  timestamp of ``input_df`` is your 1-step-ahead forecast for that hour. Use only past
  values (lags); the value at the last timestamp is intentionally NaN.

The example below simply reuses the built-in least-squares AR model as a baseline.
Replace ``MyModel`` with your own.
"""

import emflow as ef
from emflow.examples.swedish_temperatures.predictor import ARPredictor


class MyModel(ARPredictor):
    """Replace this with your own model. Here we just rename the AR baseline."""

    def __init__(self):
        super().__init__(name="intern-submission")


def get_model() -> ef.Predictor:
    """Return a fresh, untrained predictor for the verifier to train and score."""
    return MyModel()
