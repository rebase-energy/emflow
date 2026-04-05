from abc import ABC, abstractmethod
import copy

from emflow.models import Model

class Predictor(Model, ABC):
    def __init__(self):
        """
        Initialize the predictor.
        """
        pass

    def load_data(self):
        """
        Load data used for training the predictor. 

        This is a recommended method, but not mandatory. Subclasses may override it.
        """
        # Implement data loading method here. 
        pass

    def create_features(self):
        """
        Create features from the loaded data used for training the predictor.
        Load data used for training sthe predictor. 

        This is a recommended method, but not mandatory. Subclasses may override it.
        """
        # Implement feature creation method here.
        pass

    def train(self):
        """
        Train the predictor from the training data.

        This is a recommended method, but not mandatory. Subclasses may override it.
        """
        # Implement model training method here.
        pass
    
    @abstractmethod
    def predict(self, input):
        """
        Make a prediction based on the input data.

        This is a mandatory method that must be implemented by subclasses

        Parameters:
            input: Data on which prediction is to be made.

        Returns:
            prediction: The output of the model.
        """
        # Implement prediction method here. 
        pass

    def copy(self, name=None):
        """
        Create a copy of the predictor.
        """
        predictor = copy.deepcopy(self)
        predictor.name = name

        return predictor