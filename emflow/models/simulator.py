import typing as t
from abc import ABC, abstractmethod

from emflow.spaces import StateSpace
from emflow.models import Model

class Simulator(Model, ABC):
    def __init__(self, initial_state):
        """
        Initialize the Simulator with a given initial state.

        :param initial_state: The initial state of the simulator.
        """
        self.state = initial_state
        # Initialize other necessary properties

    @abstractmethod
    def _transition_function(self, state, action):
        """
        Define the state transition logic.

        :param state: The current state.
        :param action: The action taken.
        :return: The next state.
        """
        # Implement how the state changes in response to the action
        # This is where the core simulation logic is defined
        pass

    @abstractmethod
    def _gather_info(self):
        """
        Gather additional information about the current state, if necessary.

        :return: A dictionary containing additional info.
        """
        # This can include details like simulation metrics, debug info, etc.
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the simulator to the initial state or a new start state.

        :return: The reset state.
        """
        # Reset the state to its initial configuration or a new state
        pass

    @abstractmethod
    def step(self, action=None) -> t.Tuple[StateSpace, dict]:
        """
        Update the simulator's state based on an action.

        :param action: The action to be applied to the state.
        :return: A tuple containing the next state and any additional info.
        """
        next_state = self._state_transition_function(self.state, action)
        self.state = next_state
        info = self._gather_info()

        return next_state, info


