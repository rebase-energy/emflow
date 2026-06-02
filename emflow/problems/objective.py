import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class Objective(ABC):
    @abstractmethod
    def calculate(self):
        """Subclasses must implement this method."""
        pass

def _to_array(x):
    """Coerce a DataFrame/Series/list/ndarray of values to a float ndarray."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


class MeanSquaredError(Objective):
    def __init__(self):
        self._name = "MeanSquaredError"

    @property
    def name(self):
        return self._name

    def calculate(self, y_true, y_pred, mean=True, squared=True):
        """Mean squared error (or RMSE when ``squared=False``), nan-aware.

        NaN entries in either ``y_true`` or ``y_pred`` are ignored pairwise.
        Returns a scalar when ``mean=True``, else the element-wise errors.
        """
        errors = (_to_array(y_true) - _to_array(y_pred)) ** 2
        if not mean:
            return errors
        mse = np.nanmean(errors)
        return mse if squared else np.sqrt(mse)


class MeanAbsoluteError(Objective):
    def __init__(self):
        self._name = "MeanAbsoluteError"

    @property
    def name(self):
        return self._name

    def calculate(self, y_true, y_pred, mean=True):
        """Mean absolute error, nan-aware (NaNs ignored pairwise)."""
        errors = np.abs(_to_array(y_true) - _to_array(y_pred))
        return np.nanmean(errors) if mean else errors

class PinballLoss(Objective):
    def __init__(self, quantiles):
        """
        Initialize with multiple quantiles.
        :param quantiles: array-like, the quantiles for which the loss is calculated. Each must be between 0 and 1.
        """
        self.quantiles = np.array(quantiles)
        if np.any((self.quantiles <= 0) | (self.quantiles >= 1)):
            raise ValueError("Quantile values must be between 0 and 1.")

        self._name = "PinballLoss"

    @property
    def name(self):
        return self._name

    def calculate(self, y_true, y_preds, mean=True):
        """
        Compute the pinball loss between true values and multiple sets of predictions.
        Each set of predictions corresponds to a specific quantile.
        :param y_true: array-like, true values.
        :param y_preds: 2D array-like, predicted values for each quantile. Shape: (n_samples, n_quantiles).
        :return: numpy array, the pinball losses for each quantile.
        """

        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_preds, list):
            y_preds = np.array(y_preds)
        if isinstance(y_true, pd.DataFrame):
            shape = (len(y_true),) + tuple(len(y_true.columns.get_level_values(i).unique()) for i in range(y_true.columns.nlevels))
            y_true = y_true.values.reshape(shape)
        if isinstance(y_preds, pd.DataFrame):
            shape = (len(y_preds),) + tuple(len(y_preds.columns.get_level_values(i).unique()) for i in range(y_preds.columns.nlevels))
            y_preds = y_preds.values.reshape(shape)

        assert len(y_true) == y_preds.shape[0], "Number of true values must match the number of predictions."
        assert y_preds.shape[-1] == len(self.quantiles), f"Number of prediction sets {y_preds.shape[1]} must match the number of quantiles {len(self.quantiles)}."

        errors = y_true - y_preds
        losses = np.where(errors > 0, self.quantiles * errors, (self.quantiles - 1) * errors)

        if mean:
            return np.nanmean(losses)
        else:
            return losses