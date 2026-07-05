"""Simulator: the world-model contract for control/optimization problems.

A placeholder taxonomy member today: simulators become the state-transition
backends of control environments (battery dispatch, microgrids) when those
land. The interface mirrors gymnasium so a simulator can back an env directly.
"""

from abc import ABC, abstractmethod

from emflow.models import Model


class Simulator(Model, ABC):
    @abstractmethod
    def reset(self, *, seed=None, options=None):
        """Reset to an initial state; return ``(state, info)``."""

    @abstractmethod
    def step(self, action):
        """Advance one step; return ``(state, reward, terminated, truncated, info)``."""
