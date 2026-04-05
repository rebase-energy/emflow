from dataclasses import dataclass, fields
import numpy as np

from emflow.spaces import BaseSpace


@dataclass
class InputSpace(BaseSpace):
    """
    The input space for the energy model.
    """

StateSpace = InputSpace