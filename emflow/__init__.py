from .assets.base import Asset, TimeSeries, Sensor, Collection
from .assets.building import House
from .assets.solar import FixedMount, SingleAxisTrackerMount, PVArray, PVSystem, SolarPowerArea
from .assets.wind import WindTurbine, WindFarm, WindPowerArea
from .assets.battery import Battery
from .assets.heatpump import HeatPump
from .assets.energycollection import Site, EnergyCommunity, Portfolio

from .spaces.dataframe import DataFrameSpace

from .data import Dataset, DataFeed, Field, TimeView

from .models.agent import Agent
from .models.optimizer import Optimizer
from .models.predictor import Predictor
from .models.simulator import Simulator

from .problems import (
    IssueSchedule,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    Metric,
    Objective,
    Origin,
    PeakTimingError,
    PinballLoss,
    Problem,
    ProblemNotIngestedError,
    RefScore,
    RootMeanSquaredError,
    Splits,
    list_problems,
    load_problem,
)

from .envs import ForecastEnv, Observation, SettlementRecord, TradingEnv
from .problems.metrics import TradingRevenue
from .features import Calendar, ForecastField, Lag, Rolling, materialize
from .models.predictor import FeaturePredictor
from .run import (
    Analyzer,
    Experiment,
    PersistenceSkill,
    QuantileCalibration,
    Result,
    Verifier,
    default_analyzers,
    evaluate,
    load_submission,
)

# Register the Hugging Face-backed ``rb://`` fsspec filesystem so
# pandas/polars/duckdb understand it. huggingface_hub is imported lazily on
# first rb:// use, so this stays cheap and keeps the dep optional at import.
from .data.io.rebasefs import RebaseFileSystem, RebaseIncompatibleError, register as _register_rb
_register_rb()

__all__ = [
    # data
    'Dataset', 'Field', 'DataFeed', 'TimeView',
    # problems
    'Problem', 'Splits', 'RefScore', 'Objective', 'IssueSchedule', 'Origin',
    'Metric', 'MeanAbsoluteError', 'MeanSquaredError', 'RootMeanSquaredError',
    'MeanAbsolutePercentageError', 'PinballLoss', 'PeakTimingError',
    'list_problems', 'load_problem', 'ProblemNotIngestedError',
    # models
    'Predictor', 'FeaturePredictor', 'Optimizer', 'Agent', 'Simulator',
    # envs
    'ForecastEnv', 'TradingEnv', 'Observation', 'SettlementRecord',
    'TradingRevenue',
    # features
    'Lag', 'Rolling', 'ForecastField', 'Calendar', 'materialize',
    # run
    'Experiment', 'Result', 'Verifier', 'evaluate', 'load_submission',
    'Analyzer', 'PersistenceSkill', 'QuantileCalibration', 'default_analyzers',
    # io
    'RebaseFileSystem', 'RebaseIncompatibleError',
]
