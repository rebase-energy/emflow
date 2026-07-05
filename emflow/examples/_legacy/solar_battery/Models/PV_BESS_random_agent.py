import gymnasium as gym
import emflow as ef
import energydatamodel as edm
from emflow.problems.dataset import Dataset
from emflow.problems.objective import Objective

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PV_BESS_random_agent(
        database,
        EnergySystem,
        PVsyst,
        BESSsyst
):

    #Create the Dataset
    dataset = Dataset(
        name='Building 1',
        energy_system=EnergySystem,
        description='Test 1, 17/06',
        data=database
    )

    #Create spaces
    state_space = gym.spaces.Dict(
        {
            "solar_power": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "energy_consumption": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
        }
    )

    exogeneous_space = gym.spaces.Dict(
        {
            "wholesale_price": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "retail_price": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
        }
    )

    action_space = gym.spaces.Box(low=-BESSsyst.max_discharge, high=BESSsyst.max_charge, shape=(1,))

    obs_space = gym.spaces.Dict(
        {
            "BESS_SOC": gym.spaces.Box(low=BESSsyst.min_soc, high=1, shape=(1,)),
            "Trade": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),

        }
    )

    class PVenv(gym.Env):
        def __init__(
                self,
                dataset: Dataset,
        ):
            self.state_space = state_space
            self.exogeneous_space = exogeneous_space
            self.action_space = action_space
            self.observation_space = obs_space

            self.data=dataset.data
            self.portfolio = dataset.energy_system
            
            self.market_hours = self.data['Time']
            self.n_market_hours = len(self.market_hours)
            self.idx_counter = 0
            self.BESS_SOC = BESSsyst.min_soc

        def _create_state(self, idx_market_hour):

            self.exogeneous_space = {
                "wholesale_price": self.data["Wholesale prices"][idx_market_hour],
                "retail_price": self.data["Retail prices"][idx_market_hour]
            }

            self.state_space = {
                "solar_power": self.data["PV production"][idx_market_hour],
                "energy_consumption": self.data["Energy demand"][idx_market_hour]
            }

            return self.state_space, self.exogeneous_space
        
        def new_observation(self, action):
            trade = self.state_space["solar_power"] - self.state_space["energy_consumption"] - action
            #self.BESS_SOC = self.BESS_SOC + action/BESSsyst.storage_capacity

            if action>0:
                #CHARGE
                self.BESS_SOC = self.BESS_SOC + (action*BESSsyst.charge_efficiency)/BESSsyst.storage_capacity
            else:
                #DISCHARGE
                self.BESS_SOC = self.BESS_SOC + (action/BESSsyst.discharge_efficiency)/BESSsyst.storage_capacity

            self.observation_space = {
                "BESS_SOC": self.BESS_SOC,
                "Trade": trade
            }
            return self.observation_space
        
        def reset(self):
            self.idx_counter=0
            self.BESS_SOC=BESSsyst.min_soc
            initial_state, initial_exogeneous = self._create_state(self.idx_counter)
            initial_action = 0
            initial_obs = self.new_observation(initial_action)
            return initial_state, initial_exogeneous, initial_action, initial_obs

        def step(self):
            next_state, exogenous = self._create_state(self.idx_counter)
            done = True if self.idx_counter+1 == self.n_market_hours else False
            self.idx_counter += 1
            return next_state, exogenous, done

    env = PVenv(dataset=dataset)


    class obj(Objective):

        def obj_function(self, grid_trade, wholesale_price, retail_price):
            #SELLING energy to the grid
            if grid_trade>0:
                costs = - grid_trade*wholesale_price

            #BUYING energy from the grid
            if grid_trade<=0:
                costs = - grid_trade*retail_price 
            return costs
        
    calculate = obj()


    class Agent(ef.Agent):

        def act(self, state, obs):
            #EXCESS production, but the battery is full
            if state["solar_power"]>=state["energy_consumption"] and obs["BESS_SOC"]==1:
                action=0
            
            #DEFICIT, but the battery is discharged
            if state["solar_power"]<=state["energy_consumption"] and obs["BESS_SOC"]==0:
                action=0

            #CHARGE
            #action>0
            if state["solar_power"]>=state["energy_consumption"] and obs["BESS_SOC"]<1:
                action = np.random.uniform(0, min(BESSsyst.max_charge, state["solar_power"]-state["energy_consumption"], BESSsyst.storage_capacity-BESSsyst.storage_capacity*obs["BESS_SOC"]))
                
            #DISCHARGE
            #action<0
            if state["solar_power"]<=state["energy_consumption"] and obs["BESS_SOC"]>0:
                action = np.random.uniform(max(-BESSsyst.max_discharge, state["solar_power"]-state["energy_consumption"], BESSsyst.min_soc-BESSsyst.storage_capacity*obs["BESS_SOC"]), 0)

            return action

    agent = Agent()


    data = {
        'Time': pd.to_datetime([]),
        'BESS_SOC': [],
        'PV Production': [],
        'Trade': [],
        'Load': [],
        'Battery_flow': [],
        'Costs': []
    }

    df = pd.DataFrame(data)

    done = False
    initial_state, initial_exogeneous, initial_action, observation = env.reset()

    time = 0
    total_PV_Production = 0
    total_Load = 0
    total_Import = 0
    total_export = 0
    SelfSufficiency = 0
    SelfConsumption = 0

    while done is not True:
        next_state, exogeneous, done = env.step()
        action = agent.act(next_state, observation)
        observation = env.new_observation(action)
        costs = calculate.obj_function(observation["Trade"], exogeneous["wholesale_price"], exogeneous["retail_price"])


        df.loc[time, "Time"]=database.loc[time, "Time"]
        df.loc[time, "BESS_SOC"] = observation["BESS_SOC"]
        df.loc[time, "PV Production"] = next_state["solar_power"]
        df.loc[time, "Trade"] = observation["Trade"]
        df.loc[time, "Load"] = next_state["energy_consumption"]
        df.loc[time, "Battery_flow"]=action
        df.loc[time, "Costs"]=costs

        total_PV_Production += next_state["solar_power"]
        total_Load += next_state["energy_consumption"]
        if observation["Trade"]>=0:
            total_export+=observation["Trade"]
        else:
            total_Import+=observation["Trade"]

        time=time+1

    #KPI
    SolarFraction = total_PV_Production/total_Load
    SelfConsumption = (total_PV_Production-total_export)/total_PV_Production
    SelfSufficiency = (total_PV_Production-total_export)/total_Load
    data = {
        'SolarFraction': SolarFraction,
        'SelfConsumption': SelfConsumption,
        'SelfSufficiency': SelfSufficiency
    }
    KPI = pd.DataFrame(data, index=[0])

    return df, KPI



#Agent completely random
def PV_BESS_random_agent_v2(
        database,
        EnergySystem,
        PVsyst,
        BESSsyst
):

    #Create the Dataset
    dataset = Dataset(
        name='Building 1',
        energy_system=EnergySystem,
        description='Test 1, 17/06',
        data=database
    )

    #Create spaces
    state_space = gym.spaces.Dict(
        {
            "solar_power": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "energy_consumption": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
        }
    )

    exogeneous_space = gym.spaces.Dict(
        {
            "wholesale_price": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "retail_price": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
        }
    )

    action_space = gym.spaces.Box(low=-BESSsyst.max_discharge, high=BESSsyst.max_charge, shape=(1,))

    obs_space = gym.spaces.Dict(
        {
            "BESS_SOC": gym.spaces.Box(low=BESSsyst.min_soc, high=1, shape=(1,)),
            "Trade": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),

        }
    )

    class PVenv(gym.Env):
        def __init__(
                self,
                dataset: Dataset,
        ):
            self.state_space = state_space
            self.exogeneous_space = exogeneous_space
            self.action_space = action_space
            self.observation_space = obs_space

            self.data=dataset.data
            self.portfolio = dataset.energy_system
            
            self.market_hours = self.data['Time']
            self.n_market_hours = len(self.market_hours)
            self.idx_counter = 0
            self.BESS_SOC = BESSsyst.min_soc

        def _create_state(self, idx_market_hour):

            self.exogeneous_space = {
                "wholesale_price": self.data["Wholesale prices"][idx_market_hour],
                "retail_price": self.data["Retail prices"][idx_market_hour]
            }

            self.state_space = {
                "solar_power": self.data["PV production"][idx_market_hour],
                "energy_consumption": self.data["Energy demand"][idx_market_hour]
            }

            return self.state_space, self.exogeneous_space
        
        def new_observation(self, action):
            trade = self.state_space["solar_power"] - self.state_space["energy_consumption"] - action
            #self.BESS_SOC = self.BESS_SOC + action/BESSsyst.storage_capacity

            if action>0:
                #CHARGE
                self.BESS_SOC = self.BESS_SOC + (action*BESSsyst.charge_efficiency)/BESSsyst.storage_capacity
            else:
                #DISCHARGE
                self.BESS_SOC = self.BESS_SOC + (action/BESSsyst.discharge_efficiency)/BESSsyst.storage_capacity

            self.observation_space = {
                "BESS_SOC": self.BESS_SOC,
                "Trade": trade
            }
            return self.observation_space
        
        def reset(self):
            self.idx_counter=0
            self.BESS_SOC=BESSsyst.min_soc
            initial_state, initial_exogeneous = self._create_state(self.idx_counter)
            initial_action = 0
            initial_obs = self.new_observation(initial_action)
            return initial_state, initial_exogeneous, initial_action, initial_obs

        def step(self):
            next_state, exogenous = self._create_state(self.idx_counter)
            done = True if self.idx_counter+1 == self.n_market_hours else False
            self.idx_counter += 1
            return next_state, exogenous, done

    env = PVenv(dataset=dataset)


    class obj(Objective):

        def obj_function(self, grid_trade, wholesale_price, retail_price):
            #SELLING energy to the grid
            if grid_trade>0:
                costs = - grid_trade*wholesale_price

            #BUYING energy from the grid
            if grid_trade<=0:
                costs = - grid_trade*retail_price 
            return costs
        
    calculate = obj()


    class Agent(ef.Agent):

        def act(self, state, obs):
            action = np.random.choice([-BESSsyst.max_discharge, BESSsyst.max_charge])

            if obs['BESS_SOC']+action/BESSsyst.storage_capacity>1:
                action = 0

            if obs['BESS_SOC']+action/BESSsyst.storage_capacity<BESSsyst.min_soc:
                action = 0

            return action

    agent = Agent()


    data = {
        'Time': pd.to_datetime([]),
        'BESS_SOC': [],
        'PV Production': [],
        'Trade': [],
        'Load': [],
        'Battery_flow': [],
        'Costs': []
    }

    df = pd.DataFrame(data)

    done = False
    initial_state, initial_exogeneous, initial_action, observation = env.reset()

    time = 0
    total_PV_Production = 0
    total_Load = 0
    total_Import = 0
    total_export = 0
    SelfSufficiency = 0
    SelfConsumption = 0

    while done is not True:
        next_state, exogeneous, done = env.step()
        action = agent.act(next_state, observation)
        observation = env.new_observation(action)
        costs = calculate.obj_function(observation["Trade"], exogeneous["wholesale_price"], exogeneous["retail_price"])


        df.loc[time, "Time"]=database.loc[time, "Time"]
        df.loc[time, "BESS_SOC"] = observation["BESS_SOC"]
        df.loc[time, "PV Production"] = next_state["solar_power"]
        df.loc[time, "Trade"] = observation["Trade"]
        df.loc[time, "Load"] = next_state["energy_consumption"]
        df.loc[time, "Battery_flow"]=action
        df.loc[time, "Costs"]=costs

        total_PV_Production += next_state["solar_power"]
        total_Load += next_state["energy_consumption"]
        if observation["Trade"]>=0:
            total_export+=observation["Trade"]
        else:
            total_Import+=observation["Trade"]

        time=time+1

    #KPI
    SolarFraction = total_PV_Production/total_Load
    SelfConsumption = (total_PV_Production-total_export)/total_PV_Production
    SelfSufficiency = (total_PV_Production-total_export)/total_Load
    data = {
        'SolarFraction': SolarFraction,
        'SelfConsumption': SelfConsumption,
        'SelfSufficiency': SelfSufficiency
    }
    KPI = pd.DataFrame(data, index=[0])

    return df, KPI