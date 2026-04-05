class Model:
    """
    Base class for all models.

    This class provides common functionality and shared attributes for
    all derived models: :class:`Simulator`, :class:`Predictor`, :class:`Optimizer`, and :class:`Agent`.

    Attributes
    ----------
    name : str
        Name of the model.
    """
    def __init__(self, name):
        self.name = name