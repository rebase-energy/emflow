"""Submission template — copy this file, implement your model, and send it back.

The verifier will call ``get_model()`` to obtain a FRESH, UNTRAINED predictor, then:
  1. fit it on the official training view       (model.fit(train)),
  2. score it origin-by-origin on the held-out split, where every observation
     is served through emflow's point-in-time DataPortal.

You never receive the holdout targets, and your model is trained by the
verifier — so the score cannot be inflated by data leakage.

Contract for your Predictor
---------------------------
* Subclass ``emflow.Predictor`` (or ``emflow.FeaturePredictor`` for the fast
  declarative-feature path — see ``ARPredictor`` for a full example).
* ``fit(self, train)``: ``train`` is a TimeView — pull the training series with
  ``train.history("temperature")``. Use only what it serves.
* ``predict(self, obs)``: return a DataFrame indexed by ``obs.target_index``
  with a ``"point"`` column. ``obs.history("temperature")`` gives you
  everything knowable at the origin — the targets are not in it.

Self-check before submitting::

    python scripts/verify_submission.py submissions/template_submission.py

The example below reuses the built-in least-squares AR baseline. Replace
``MyModel`` with your own.
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
