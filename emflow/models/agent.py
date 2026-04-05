from abc import ABC, abstractmethod
from emflow.models import Model

class Agent(Model, ABC):
    def __init__(self):
        # Initialize agent properties
        pass
    
    @abstractmethod
    def act(self, state):
        """
        Choose an action based on the given state.

        Parameters:
            state: The current state of the environment.

        Returns:
            action: The action chosen by the agent.
        """
        # Implement logic to choose an action
        # For example, a random choice:
        # return env.action_space.sample()
        pass

    def learn(self, state, action, reward, next_state):
        """
        Learn from the experience (used in training).

        Parameters:
            state: The current state of the environment.
            action: The action taken in the state.
            reward: The reward received from the environment.
            next_state: The next state of the environment.
        """
        # Implement learning process
        pass
