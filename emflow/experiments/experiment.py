from abc import ABC, abstractmethod


class Experiment:
    def __init__(self, problem, model):
        """
        Initialize the experiment.
        """
        self.problem = problem
        self.model = model

    def run(self):
        """
        Run the experiment.
        """
        env = self.problem.environment
        state = env.reset()
        objective = self.problem.objective
        done = False

        rewards = []
        while not done:
            action = self.model.get_action(state)
            state, done, info = env.step(action)
            reward = objective.evaluate(state, action, info)
            rewards.append(reward)