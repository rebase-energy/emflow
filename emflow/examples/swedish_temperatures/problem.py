"""Swedish temperature forecasting problem (Stockholm, 1-step-ahead AR).

Defines an emflow example problem over the hourly Swedish temperature dataset,
scoped initially to the Stockholm-Observatoriekullen A station (id ``98230``).
The task: train on all data before 2026 and forecast 2026 one hour ahead.

Discoverable via ``emflow.load_problem("swedish-temperatures:ar")``.
"""

import importlib.resources

import numpy as np
import pandas as pd
import gymnasium as gym

import emflow as ef
from emflow.problems.objective import MeanAbsoluteError

# --- Configuration ----------------------------------------------------------

STATION_ID = "98230"
STATION_NAME = "Stockholm-Observatoriekullen A"
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")  # train < TEST_START <= test

# --- Load data --------------------------------------------------------------

with importlib.resources.path("emflow.examples.data", "swedish-temperatures.parquet") as path:
    _df = pd.read_parquet(path)
with importlib.resources.path("emflow.examples.data", "swedish-temperatures-stations.parquet") as path:
    _meta = pd.read_parquet(path)

# Keep only the Stockholm column (preserving the (station_id, station_name) MultiIndex).
df_temperatures = _df.xs(STATION_ID, level="station_id", axis=1, drop_level=False)

# --- Energy collection (one weather-station sensor) -------------------------
# Station coordinates live in dataset.data["stations"]; the sensor is kept
# name-only so the example does not depend on a specific energydatamodel
# geolocation API.

sensor = ef.Sensor(name=STATION_NAME)
portfolio = ef.Portfolio(name="SwedishStations", members=[sensor])

dataset = ef.Dataset(
    name="swedish-temperatures",
    description=(
        "Hourly air temperature (°C) from SMHI station "
        f"{STATION_NAME} (id {STATION_ID}). Benchmark for 1-step-ahead "
        "autoregressive forecasting: train < 2026, evaluate on 2026."
    ),
    collection=portfolio,
    data={"temperatures": df_temperatures, "stations": _meta},
)

# --- Spaces -----------------------------------------------------------------

state_space = ef.DataFrameSpace({
    STATION_NAME: {
        "temperature": gym.spaces.Box(low=-60.0, high=50.0, shape=(1,), dtype=np.float32),
    }
})
action_space = ef.DataFrameSpace({
    STATION_NAME: {
        "temperature_forecast": gym.spaces.Box(low=-60.0, high=50.0, shape=(1,), dtype=np.float32),
    }
})


# --- Environment ------------------------------------------------------------

class SwedishTemperatureEnv(gym.Env):
    """Single-window walk-forward env for 1-step-ahead temperature forecasting.

    ``reset()`` hands back the pre-2026 training series and the full series as the
    forecast input (the AR predictor uses true lags, so it may see the whole
    series; predictions are scored only on the 2026 test window). ``step()``
    returns the 2026 actuals to evaluate against.
    """

    def __init__(self, dataset, state_space, action_space,
                 test_start=TEST_START):
        self.dataset = dataset
        self.temps = dataset.data["temperatures"]
        self.test_start = pd.Timestamp(test_start)
        self.state_space = state_space
        self.action_space = action_space
        self.n_steps = 1
        self.idx_counter = 0

    def reset(self):
        self.idx_counter = 0
        train_temps = self.temps[self.temps.index < self.test_start]
        # Full series as input: 1-step-ahead AR consumes true observed lags; the
        # runner aligns predictions to the test window via reindex.
        first_input = self.temps
        initial_data = {"input": train_temps, "target": train_temps}
        return initial_data, first_input

    def step(self, action=None):
        test_target = self.temps[self.temps.index >= self.test_start]
        done = True
        self.idx_counter += 1
        return None, test_target, done


# --- Problem factory --------------------------------------------------------

def list_problem_variants():
    return ["ar"]


def get_problem_ar():
    env = SwedishTemperatureEnv(
        dataset=dataset, state_space=state_space, action_space=action_space,
    )
    objective = MeanAbsoluteError()
    ef.Problem(name="swedish-temperatures:ar", environment=env, objective=objective)
    return dataset, env, objective
