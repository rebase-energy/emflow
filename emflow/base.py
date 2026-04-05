import typing as t
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, fields
import numpy as np

import pandas as pd
import energydatamodel as edm


@dataclass
class BaseVector:
    """
    Base vector.
    """

    @property
    def vector(self) -> np.ndarray:
        """ 
        Get all fields (attributes) of the dataclass as a vector. 
        """

        all_fields = fields(self)        
        values = [getattr(self, field.name) for field in all_fields]
        array = np.array(values)
        return array

    @classmethod
    def from_vector(cls, state_vector):
        """ 
        Create the :class:BaseVector from a :class:numpy.array. 
        """

        attribute_names = [field.name for field in fields(cls)]

        if len(state_vector) != len(attribute_names):
            raise ValueError("State vector and attribute names must have the same length")

        attribute_values = dict(zip(attribute_names, state_vector))
        state = cls(**attribute_values)
        return state


@dataclass
class BaseState(BaseVector):
    pass


@dataclass
class BaseAction(BaseVector):
    pass


@dataclass
class BaseEnvironment(gym.Env):
    pass


@dataclass
class BaseObjective:
    pass


@dataclass
class BaseProblem:
    collection: edm.EnergyCollection
    state: BaseState
    action: BaseAction
    environment: BaseEnvironment
    objective: BaseObjective

