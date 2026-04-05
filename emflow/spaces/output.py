from dataclasses import dataclass, fields
import numpy as np

from emflow.spaces import BaseSpace


@dataclass
class OutputSpace(BaseSpace):
    """
    The output space for the energy model.
    """

ActionSpace = OutputSpace