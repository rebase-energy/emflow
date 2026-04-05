import gymnasium
import pandas as pd
import numpy as np
from gymnasium.spaces import Space, Dict, Box

class DataFrameSpace(Space):
    def __init__(self, space_dict):
        """
        Initialize the DataFrameSpace with a dictionary or gym.spaces.Dict.
        
        :param space_dict: A Python dictionary or gym.spaces.Dict where each key maps to a space for a column.
                           A nested dictionary will be converted to a two-level multiindex column.
        """
        #TODO: make it possible to pass a spaces.Box with a shape of (n,) to create n columns
          
        assert isinstance(space_dict, (dict, Dict)), "Input must be a Python dict or gym.spaces.Dict."
        
        # Convert any nested dicts into gym.spaces.Dict
        self.space_dict = self._convert_to_space_dict(space_dict)
        
        # Build the column structure from the dict, handling potential nested dicts
        self.columns = pd.MultiIndex.from_tuples(self._build_columns(self.space_dict))
        
        # Call the parent class constructor
        super().__init__(shape=None, dtype=None)

    def _convert_to_space_dict(self, space_dict):
        """
        Convert nested Python dictionaries into gym.spaces.Dict.
        """
        for key, value in space_dict.items():
            if isinstance(value, dict):
                space_dict[key] = Dict(self._convert_to_space_dict(value))
        return Dict(space_dict)

    def _build_columns(self, space_dict, parent_key=()):
        """
        Recursively build a list of column names (or tuples for multiindex) from the input dictionary.
        """
        columns = []
        for key, space in space_dict.spaces.items():
            if isinstance(space, Dict):
                # If the value is a Dict, recursively add to columns
                sub_columns = self._build_columns(space, parent_key + (key,))
                columns.extend(sub_columns)
            else:
                # Otherwise, just add the current key to columns
                columns.append(parent_key + (key,))
        return columns

    def sample(self, n_rows=None, index=None):
        """
        Generate a sample from the space.
        
        :param n_rows: Number of rows to generate for the DataFrame.
        :param index: An optional pandas Index or MultiIndex to use for the DataFrame.
        :return: A pandas DataFrame with sampled values.
        """
        if index is not None:
            assert isinstance(index, (pd.Index, pd.MultiIndex)), "Index must be a pandas Index or MultiIndex."
            n_rows = len(index)
        elif n_rows is None:
            n_rows = 1  # Default to 1 row if neither n_rows nor index are provided
        
        # Sample values for each column based on the gym space
        data = {col: self._sample_from_space(self.space_dict, col, n_rows) for col in self.columns}

        # Create the DataFrame with the sampled data
        df = pd.DataFrame(data, columns=self.columns, index=index)
        return df

    def _sample_from_space(self, space_dict, col, n_rows):
        """
        Traverse the space_dict to reach the final space (e.g., Box, Discrete) and sample from it.
        
        :param space_dict: The gym.spaces.Dict or space container.
        :param col: A tuple representing the multi-level column.
        :param n_rows: The number of rows to sample.
        :return: A sampled array from the corresponding space.
        """
        space = space_dict
        for key in col:
            if isinstance(space, Dict):
                space = space.spaces[key]  # Traverse the space_dict using the column keys
            else:
                break
            
        # Now that space is a final gym.Space (e.g., Box, Discrete, etc.), we can sample from it
        return np.array([space.sample().squeeze() for _ in range(n_rows)])

    def contains(self, x):
        """
        Check if a given dataframe x is contained within the space.
        
        :param x: A pandas DataFrame.
        :return: True if x is contained in the space, False otherwise.
        """
        if not isinstance(x, pd.DataFrame):
            return False
        
        # Check if all columns are present
        if not all(col in x.columns for col in self.columns):
            return False
        
        # Check if each column's values lie within the space
        for col in self.columns:
            space = self.space_dict
            for key in col:
                space = space.spaces[key]
            if isinstance(space, Dict):
                # Recursively check Dict spaces
                if not all(self._contains_space(space, x[subcol]) for subcol in x.columns):
                    return False
            else:
                # Check using the space's contains method
                if not all(space.contains(val) for val in x[col]):
                    return False
        
        return True

    def _contains_space(self, space, x):
        """
        Recursively check if the space contains the values of a sub-space.
        """
        if isinstance(space, Dict):
            # Recursively check Dict spaces
            return all(self._contains_space(sub_space, x[subcol]) for subcol, sub_space in space.spaces.items())
        else:
            # Use the space's contains method for non-Dict spaces
            return all(space.contains(val) for val in x)

    def __repr__(self):
        return f"DataFrameSpace({self.space_dict})"
