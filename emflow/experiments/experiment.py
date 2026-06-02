"""Run a predictor against an emflow problem (the standard walk-forward loop).

This is the plain evaluation runner used in the example notebooks: reset the
environment, train on the initial window, then for each step predict, advance the
environment to reveal the targets, and score with the problem's objective. For
*untrusted* submissions where leakage must be impossible, use
:class:`emflow.experiments.verifier.Verifier` instead.
"""

import numpy as np


class Experiment:
    def __init__(self, problem, model):
        """Initialize the experiment with a Problem and a Predictor."""
        self.problem = problem
        self.model = model

    def run(self):
        """Run the reset → train → (predict → step → score)* loop.

        Returns a dict with the per-step losses and their mean.
        """
        env = self.problem.environment
        objective = self.problem.objective

        initial_data, next_input = env.reset()
        self.model.train(initial_data["target"])

        losses, predictions = [], []
        done = False
        while not done:
            prediction = self.model.predict(next_input)
            predictions.append(prediction)
            next_input, target, done = env.step()
            if target is not None:
                aligned = prediction.reindex(target.index) if hasattr(prediction, "reindex") else prediction
                losses.append(float(objective.calculate(target, aligned)))

        return {
            "model": getattr(self.model, "name", type(self.model).__name__),
            "objective": objective.name,
            "losses": losses,
            "mean_loss": float(np.nanmean(losses)) if losses else float("nan"),
        }
