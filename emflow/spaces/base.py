from dataclasses import dataclass, fields
import numpy as np

@dataclass
class BaseSpace:
    """
    A base space with methods for converting data formats.

    """

    @classmethod
    def from_array(cls, input_array: np.ndarray) -> "BaseSpace":
        """
        Create the action from a numpy array.
        """
        action = cls(*input_array)
        return action

    def to_array(self) -> np.ndarray:
        """
        Convert the action to a numpy array.
        """
        all_fields = fields(self)        
        values = [getattr(self, field.name) for field in all_fields]
        output_array = np.array(values)
        return output_array

    @property
    def vector(self) -> np.ndarray:
        return self.to_array()

    @classmethod
    def from_tuple(cls, input_tuple):
        """
        Create the action from a tuple.
        """        
        return cls(*input_tuple)

    def to_tuple(self):
        """
        Convert the action to a tuple.
        """       
        all_fields = fields(self)        
        values = [getattr(self, field.name) for field in all_fields]
        output_tuple = tuple(values)
        return output_tuple

    @property
    def tuple(self) -> tuple:
        return self.to_tuple()