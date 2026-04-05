"""This example showcases how to use emflow for the solar forecasting task of GEFCom2014."""
import sys, os
sys.path.insert(0, os.path.abspath('../../../'))

import emflow
import emflow as emf

import gym
from gym import spaces
from dataclasses import dataclass
import numpy as np

import energydatamodel as edm

# Problems: 
# 1) How do I define an initial state with a different size compared to the consecutive states? 
# 2) How do I provide datetime index information as part of the state? Should I provide a pandas as the state information instead of numpy? 

def define_system(): 
    timeseries_1 = edm.TimeSeries(df=df, column_name=('1', 'POWER'))
    pvsystem_1 = edm.PVArray(capacity=1,
                            longitude=145,
                            latitude=-37.5,
                            altitude=595,
                            surface_azimuth=38,
                            surface_tilt=36,
                            timeseries=timeseries_1)

    timeseries_2 = edm.TimeSeries(df=df, column_name=('2', 'POWER'))
    pvsystem_2 = edm.PVArray(capacity=1,
                            longitude=145,
                            latitude=-37.5,
                            altitude=602,
                            surface_azimuth=327,
                            surface_tilt=35,
                            timeseries=timeseries_2)

    timeseries_3 = edm.TimeSeries(df=df, column_name=('3', 'POWER'))
    pvsystem_3 = edm.PVArray(capacity=1,
                            longitude=145,
                            latitude=-37.5,
                            altitude=951,
                            surface_azimuth=31,
                            surface_tilt=21,
                            timeseries=timeseries_3)

    portfolio = edm.Portfolio(assets=[pvsystem_1, pvsystem_2, pvsystem_3])

    return portfolio


@dataclass
class State(emf.BaseState):
    """
    The state vector.

    Args:
        pv_production [kWh/h]: Power production from solar wind used in the microgrid.
    """

    pv_production_1: gym.spaces.Box
    pv_production_2: gym.spaces.Box
    pv_production_3: gym.spaces.Box


@dataclass
class Action:
    test: float = 4

class GEFCom2014SolarEnv(gym.Env):
    """An Environment for GEFCom2014 Solar Competition."""
    metadata = {'render.modes': ['human']}

    def __init__(self, portfolio, df):
        super(GEFCom2014SolarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.portfolio = portfolio
        self.df = df
        self.initial_state_space = self._define_initial_state_space()
        self.state_space = self._define_task_state_space()
        self.action_space = self._define_task_state_space()

    def step(self, action):
        # Execute one time step within the environment
        # return observation, reward, done, info
        return 0

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = self._set_initial_state()
        return self._get_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def close(self):
        # Clean up resources
        pass

    def _define_initial_state_space(self):
        initial_start = "2012-04-01 01:00:00"
        initial_end = "2013-04-01 00:00:00"
        initial_datapoints = len(self.df[initial_start:initial_end])
        initial_state = self._define_state_space(initial_datapoints)
        return initial_state

    def _define_task_state_space(self):
        task1_start = "2013-04-01 01:00:00"       
        task1_end = "2013-04-01 01:00:00"       
        task_datapoints = len(self.df[task1_start:task1_end])
        task_state = self._define_state_space(task_datapoints)
        return task_state

    def _define_state_space(self, datapoints):
        pv_production_1 = spaces.Box(low=0, high=self.portfolio.assets[0].capacity, shape=(datapoints, 1), dtype=np.float32)
        pv_production_2 = spaces.Box(low=0, high=self.portfolio.assets[1].capacity, shape=(datapoints, 1), dtype=np.float32)
        pv_production_3 = spaces.Box(low=0, high=self.portfolio.assets[2].capacity, shape=(datapoints, 1), dtype=np.float32)
        state = spaces.Tuple((pv_production_1, pv_production_2, pv_production_3))
        return state
    
    def _get_observation(self):
        return self.state