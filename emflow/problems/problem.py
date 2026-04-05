from dataclasses import dataclass
import typing as t

import energydatamodel as edm
import gymnasium as gym
from .objective import Objective

@dataclass
class Problem:
    name: str
    environment: gym.Env
    objective: Objective
    description: t.Optional[str] = None
