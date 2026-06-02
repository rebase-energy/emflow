from .assets.base import Asset, TimeSeries, Sensor, Collection
from .assets.building import House
from .assets.solar import FixedMount, SingleAxisTrackerMount, PVArray, PVSystem, SolarPowerArea
from .assets.wind import WindTurbine, WindFarm, WindPowerArea
from .assets.battery import Battery
from .assets.heatpump import HeatPump
from .assets.energycollection import Site, EnergyCommunity, Portfolio

from .spaces.base import BaseSpace
from .spaces.input import InputSpace, StateSpace
from .spaces.output import OutputSpace, ActionSpace
from .spaces.dataframe import DataFrameSpace

from .models.agent import Agent
from .models.optimizer import Optimizer
from .models.predictor import Predictor
from .models.simulator import Simulator

from .problems.dataset import Dataset
from .problems.objective import Objective
from .problems.problem import Problem

from .problems.objective import PinballLoss
#from energydatamodel.pv import PVArray

from .utils.loader import list_problems, load_problem

from .experiments.experiment import Experiment
from .experiments.verifier import Verifier

# Register the Hugging Face-backed ``rb://`` fsspec filesystem so
# pandas/polars/duckdb understand it. huggingface_hub is imported lazily on
# first rb:// use, so this stays cheap and keeps the dep optional at import.
from .io.rebasefs import RebaseFileSystem, RebaseIncompatibleError, register as _register_rb
_register_rb()

__all__ = ['list_problems', 'load_problem', 'Experiment', 'Verifier',
           'RebaseFileSystem', 'RebaseIncompatibleError']
