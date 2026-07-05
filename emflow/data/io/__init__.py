"""IO integrations for emflow, including the ``rb://`` fsspec filesystem."""

from .grid import fetch_series, grid_field, load_series
from .rebasefs import RebaseFileSystem, RebaseIncompatibleError, register

__all__ = ["RebaseFileSystem", "RebaseIncompatibleError", "register",
           "fetch_series", "grid_field", "load_series"]
