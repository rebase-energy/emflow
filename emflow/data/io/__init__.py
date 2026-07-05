"""IO integrations for emflow, including the ``rb://`` fsspec filesystem."""

from .rebasefs import RebaseFileSystem, RebaseIncompatibleError, register

__all__ = ["RebaseFileSystem", "RebaseIncompatibleError", "register"]
