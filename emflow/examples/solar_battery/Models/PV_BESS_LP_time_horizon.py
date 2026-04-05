import gymnasium as gym
import emflow as ef
import energydatamodel as edm
from emflow.problems.dataset import Dataset
from emflow.problems.objective import Objective

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib

from Utilities.PV_AC_Output import PV_AC_Output
from Utilities.Data_import import import_load_from_CSV, calculate_retail_price, import_from_Excel

from Models.PV_BESS_LP_opt import PV_BESS_LP_solver



def PV_BESS_LP_Time_horizon_v2(
        EnergySystem,
        database,
        time_horizon = 24
):
    PVsyst = EnergySystem.assets[0]
    BESSsyst = EnergySystem.assets[1]
    
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
            #Capire come impostare time in formato data
            "time": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "solar_power": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "energy_consumption": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
        }
    )

    exogeneous_space = gym.spaces.Dict(
        {
            "wholesale_price": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "retail_price": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
        }
    )

    action_space = gym.spaces.Box(low=-BESSsyst.max_discharge*BESSsyst.storage_capacity, high=BESSsyst.max_charge*BESSsyst.storage_capacity, shape=(time_horizon,))

    obs_space = gym.spaces.Dict(
        {
            "BESS_SOC": gym.spaces.Box(low=0, high=BESSsyst.storage_capacity, shape=(1,)),
            "Trade": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),

        }
    )


    class Predictor(ef.Predictor):
        def __init__(
                self
        ):
            self.EnergySystem = EnergySystem
            self.PVsyst = PVsyst
            self.BESSsyst = BESSsyst

        def predict(self, data_time_horizon, initial_SOC):
            
            df_time_horizon, _ = PV_BESS_LP_solver(data_time_horizon, 
                                self.EnergySystem, 
                                initial_soc=initial_SOC/BESSsyst.storage_capacity)

            return df_time_horizon
        
    model = Predictor()


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
            self.n_market_hours = len(self.market_hours)-time_horizon
            self.idx_counter = 0
            self.BESS_SOC = BESSsyst.min_soc*BESSsyst.storage_capacity
            self.time_horizon = time_horizon

        def _create_state(self, idx_market_hour):

            self.exogeneous_space = {
                "wholesale_price": self.data["WholesalePrices"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True),
                "retail_price": self.data["RetailPrices"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True)
            }

            self.state_space = {
                "time": self.data["Time"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True),
                "solar_power": self.data["PVProduction"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True),
                "energy_consumption": self.data["EnergyDemand"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True)
            }

            return self.state_space, self.exogeneous_space
        
        def new_observation(self, action):
            trade = self.state_space["solar_power"].loc[0] - self.state_space["energy_consumption"].loc[0] - action

            self.BESS_SOC += action

            self.observation_space = {
                "BESS_SOC": self.BESS_SOC,
                "Trade": trade
            }
            return self.observation_space
        
        def last_observation(self, action):
            i = self.idx_counter - self.n_market_hours

            trade = self.state_space["solar_power"].loc[i] - self.state_space["energy_consumption"].loc[i] - action

            self.BESS_SOC += action

            self.observation_space = {
                "BESS_SOC": self.BESS_SOC,
                "Trade": trade
            }
            
            self.idx_counter+=1 
            
            return self.observation_space
        
        def reset(self):
            self.idx_counter=0
            self.BESS_SOC=BESSsyst.min_soc*BESSsyst.storage_capacity
            initial_state, initial_exogeneous = self._create_state(self.idx_counter)
            initial_action = 0
            initial_obs = self.new_observation(initial_action)
            return initial_state, initial_exogeneous, initial_action, initial_obs

        def step(self):
            next_state, exogenous = self._create_state(self.idx_counter)
            done = True if self.idx_counter == self.n_market_hours else False
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
        def act(self, state, exogeneous, initial_soc):

            data = {
                "Time": state["time"],
                "EnergyDemand": state["energy_consumption"],
                "PVProduction": state["solar_power"],
                "Wholesale prices": exogeneous["wholesale_price"],
                "Retail prices": exogeneous["retail_price"]
            }

            data_time_horizon = pd.DataFrame(data)
            data_time_horizon.reset_index(drop=True, inplace=True)
            df = model.predict(data_time_horizon, initial_soc)

            action = df['BatteryFlow'].loc[0]

            return action
        
        def last_act(self, state, exogeneous, initial_soc):
            data = {
                "Time": state["time"],
                "EnergyDemand": state["energy_consumption"],
                "PVProduction": state["solar_power"],
                "Wholesale prices": exogeneous["wholesale_price"],
                "Retail prices": exogeneous["retail_price"]
            }

            data_time_horizon = pd.DataFrame(data)
            data_time_horizon.reset_index(drop=True, inplace=True)
            df = model.predict(data_time_horizon, initial_soc)

            action = df['BatteryFlow']

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

    time = env.idx_counter
    total_PV_Production = 0
    total_Load = 0
    total_Import = 0
    total_export = 0
    SelfSufficiency = 0
    SelfConsumption = 0


    while done is not True:
        next_state, exogeneous, done = env.step()

        if done:
            action = agent.last_act(next_state, exogeneous, observation['BESS_SOC'])
            for i in range(len(next_state["time"])):
                if i==0:
                    observation = env.new_observation(action[i])
                else:
                    observation = env.last_observation(action[i])

                costs = calculate.obj_function(observation["Trade"], exogeneous["wholesale_price"].loc[i], exogeneous["retail_price"].loc[i])

                df.loc[time, "Time"]=next_state["time"].loc[i]
                df.loc[time, "BESS_SOC"] = observation["BESS_SOC"]
                df.loc[time, "PV Production"] = next_state["solar_power"].loc[i]
                df.loc[time, "Trade"] = observation["Trade"]
                df.loc[time, "Load"] = next_state["energy_consumption"].loc[i]
                df.loc[time, "Battery_flow"]=action[i]
                df.loc[time, "Costs"]=costs
                
                total_PV_Production += next_state["solar_power"].loc[i]
                total_Load += next_state["energy_consumption"].loc[i]
                if observation["Trade"]>=0:
                    total_export+=observation["Trade"]
                else:
                    total_Import+=observation["Trade"]

                time=env.idx_counter
        else:
            action = agent.act(next_state, exogeneous, observation['BESS_SOC'])
            observation = env.new_observation(action)
            costs = calculate.obj_function(observation["Trade"], exogeneous["wholesale_price"].loc[0], exogeneous["retail_price"].loc[0])


            df.loc[time, "Time"]=next_state["time"].loc[0]
            df.loc[time, "BESS_SOC"] = observation["BESS_SOC"]
            df.loc[time, "PV Production"] = next_state["solar_power"].loc[0]
            df.loc[time, "Trade"] = observation["Trade"]
            df.loc[time, "Load"] = next_state["energy_consumption"].loc[0]
            df.loc[time, "Battery_flow"]=action
            df.loc[time, "Costs"]=costs
            
            total_PV_Production += next_state["solar_power"].loc[0]
            total_Load += next_state["energy_consumption"].loc[0]
            if observation["Trade"]>=0:
                total_export+=observation["Trade"]
            else:
                total_Import+=observation["Trade"]

            time=env.idx_counter

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









def PV_BESS_LP_Time_horizon(
        EnergySystem,
        PVsyst,
        BESSsyst,
        database,
        time_horizon = 24
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
            #Capire come impostare time in formato data
            "time": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "solar_power": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "energy_consumption": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
        }
    )

    exogeneous_space = gym.spaces.Dict(
        {
            "wholesale_price": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "retail_price": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
        }
    )

    action_space = gym.spaces.Box(low=-BESSsyst.max_discharge, high=BESSsyst.max_charge, shape=(time_horizon,))

    obs_space = gym.spaces.Dict(
        {
            "BESS_SOC": gym.spaces.Box(low=0, high=BESSsyst.storage_capacity, shape=(time_horizon,)),
            "Trade": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(time_horizon,)),

        }
    )


    class Predictor(ef.Predictor):
        def __init__(
                self
        ):
            self.EnergySystem = EnergySystem
            self.PVsyst = PVsyst
            self.BESSsyst = BESSsyst

        def predict(self, data_time_horizon, initial_SOC):
            
            df_time_horizon, _ = PV_BESS_LP_solver(data_time_horizon, 
                                self.EnergySystem, 
                                self.PVsyst, 
                                self.BESSsyst,
                                initial_soc=initial_SOC)
            
            #battery_flow_time_horizon = df_time_horizon['Battery_flow']

            return df_time_horizon
        
    model = Predictor()



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
            self.BESS_SOC = BESSsyst.min_soc*np.ones(time_horizon)
            self.time_horizon = time_horizon

        def _create_state(self, idx_market_hour):

            self.exogeneous_space = {
                "wholesale_price": self.data["Wholesale prices"][idx_market_hour:idx_market_hour+self.time_horizon],
                "retail_price": self.data["Retail prices"][idx_market_hour:idx_market_hour+self.time_horizon]
            }

            self.state_space = {
                "time": self.data["Time"][idx_market_hour:idx_market_hour+self.time_horizon],
                "solar_power": self.data["PV production"][idx_market_hour:idx_market_hour+self.time_horizon],
                "energy_consumption": self.data["Energy demand"][idx_market_hour:idx_market_hour+self.time_horizon]
            }

            return self.state_space, self.exogeneous_space
        
        def new_observation(self, action, new_SOC):
            trade = self.state_space["solar_power"].reset_index(drop=True) - self.state_space["energy_consumption"].reset_index(drop=True) - action

            initial_soc_next_state = new_SOC[self.time_horizon-1]

            self.observation_space = {
                "BESS_SOC": new_SOC,
                "Trade": trade
            }
            return self.observation_space, initial_soc_next_state
        
        def reset(self):
            self.idx_counter=0
            self.BESS_SOC=BESSsyst.min_soc*np.ones(time_horizon)
            initial_obs = self.BESS_SOC
            initial_state, initial_exogeneous = self._create_state(self.idx_counter)
            initial_action = np.zeros(time_horizon)
            
            initial_soc_next_state=BESSsyst.min_soc
            

            return initial_state, initial_exogeneous, initial_action, initial_obs, initial_soc_next_state

        def step(self):
            next_state, exogenous = self._create_state(self.idx_counter)
            done = True if self.idx_counter+time_horizon == self.n_market_hours else False
            self.idx_counter += time_horizon
            return next_state, exogenous, done

    env = PVenv(dataset=dataset)


    class obj(Objective):

        def obj_function(self, grid_trade, wholesale_price, retail_price):
            costs=np.zeros(len(grid_trade))
            for i in range(len(grid_trade)):
                #SELLING energy to the grid
                if grid_trade[i]>0:
                    costs[i] = - grid_trade[i]*wholesale_price[i]

                #BUYING energy from the grid
                if grid_trade[i]<=0:
                    costs[i] = - grid_trade[i]*retail_price[i] 
            return costs
        
    calculate = obj()


    class Agent(ef.Agent):

        def act(self, state, exogeneous, initial_soc):

            data = {
                "Time": state["time"],
                "Energy demand": state["energy_consumption"],
                "PV production": state["solar_power"],
                "Wholesale prices": exogeneous["wholesale_price"],
                "Retail prices": exogeneous["retail_price"]
            }

            data_time_horizon = pd.DataFrame(data)
            data_time_horizon.reset_index(drop=True, inplace=True)
            df = model.predict(data_time_horizon, initial_soc)

            action = df['Battery_flow']
            new_soc = df['BESS_SOC']

            return action, new_soc

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
    initial_state, initial_exogeneous, initial_action, observation, initial_soc_next_state = env.reset()
    time = 0

    total_PV_Production = 0
    total_Load = 0
    total_Import = 0
    total_export = 0
    SelfSufficiency = 0
    SelfConsumption = 0

    while done is not True:
        next_state, exogeneous, done = env.step()
        action, new_soc = agent.act(next_state, exogeneous, initial_soc=initial_soc_next_state)
        observation, initial_soc_next_state = env.new_observation(action, new_soc)
        costs = calculate.obj_function(observation["Trade"], exogeneous["wholesale_price"].reset_index(drop=True), exogeneous["retail_price"].reset_index(drop=True))

        data = {
        'Time': pd.to_datetime([]),
        'BESS_SOC': [],
        'PV Production': [],
        'Trade': [],
        'Load': [],
        'Battery_flow': [],
        'Costs': []
        }

        ResultsDF = pd.DataFrame(data)

        ResultsDF["Time"]=next_state["time"].reset_index(drop=True)
        ResultsDF["BESS_SOC"] = observation["BESS_SOC"]
        ResultsDF["PV Production"] = next_state["solar_power"].reset_index(drop=True)
        ResultsDF["Trade"] = observation["Trade"]
        ResultsDF["Load"] = next_state["energy_consumption"].reset_index(drop=True)
        ResultsDF["Battery_flow"]=action
        ResultsDF["Costs"]=costs

        df = pd.concat([df, ResultsDF], ignore_index=True)

        total_PV_Production += sum(next_state["solar_power"])
        total_Load += sum(next_state["energy_consumption"])

        for i in range(len(observation["Trade"])):
            if observation["Trade"].loc[i]>=0:
                total_export+=observation["Trade"].loc[i]
            else:
                total_Import+=observation["Trade"].loc[i]
        
        print(time)
        time=time+time_horizon

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




def PV_BESS_LP_Time_horizon_v2_OLD( #Not working anymore
        EnergySystem,
        PVsyst,
        BESSsyst,
        database,
        time_horizon = 24
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
            #Capire come impostare time in formato data
            "time": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "solar_power": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "energy_consumption": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
        }
    )

    exogeneous_space = gym.spaces.Dict(
        {
            "wholesale_price": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
            "retail_price": gym.spaces.Box(low=0, high=np.inf, shape=(time_horizon,)),
        }
    )

    action_space = gym.spaces.Box(low=-BESSsyst.max_discharge*BESSsyst.storage_capacity, high=BESSsyst.max_charge*BESSsyst.storage_capacity, shape=(time_horizon,))

    obs_space = gym.spaces.Dict(
        {
            "BESS_SOC": gym.spaces.Box(low=0, high=BESSsyst.storage_capacity, shape=(1,)),
            "Trade": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),

        }
    )


    class Predictor(ef.Predictor):
        def __init__(
                self
        ):
            self.EnergySystem = EnergySystem
            self.PVsyst = PVsyst
            self.BESSsyst = BESSsyst

        def predict(self, data_time_horizon, initial_SOC):
            
            df_time_horizon, _ = PV_BESS_LP_solver(data_time_horizon, 
                                self.EnergySystem, 
                                initial_soc=initial_SOC/BESSsyst.storage_capacity)

            return df_time_horizon
        
    model = Predictor()


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
            self.n_market_hours = len(self.market_hours)-time_horizon
            self.idx_counter = 0
            self.BESS_SOC = BESSsyst.min_soc*BESSsyst.storage_capacity
            self.time_horizon = time_horizon

        def _create_state(self, idx_market_hour):

            self.exogeneous_space = {
                "wholesale_price": self.data["Wholesale prices"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True),
                "retail_price": self.data["Retail prices"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True)
            }

            self.state_space = {
                "time": self.data["Time"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True),
                "solar_power": self.data["PV production"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True),
                "energy_consumption": self.data["Energy demand"][idx_market_hour:idx_market_hour+self.time_horizon].reset_index(drop=True)
            }

            return self.state_space, self.exogeneous_space
        
        def new_observation(self, action):
            trade = self.state_space["solar_power"].loc[0] - self.state_space["energy_consumption"].loc[0] - action

            self.BESS_SOC += action

            self.observation_space = {
                "BESS_SOC": self.BESS_SOC,
                "Trade": trade
            }
            return self.observation_space
        
        def last_observation(self, action):
            i = self.idx_counter - self.n_market_hours

            trade = self.state_space["solar_power"].loc[i] - self.state_space["energy_consumption"].loc[i] - action

            self.BESS_SOC += action

            self.observation_space = {
                "BESS_SOC": self.BESS_SOC,
                "Trade": trade
            }
            
            self.idx_counter+=1 
            
            return self.observation_space
        
        def reset(self):
            self.idx_counter=0
            self.BESS_SOC=BESSsyst.min_soc*BESSsyst.storage_capacity
            initial_state, initial_exogeneous = self._create_state(self.idx_counter)
            initial_action = 0
            initial_obs = self.new_observation(initial_action)
            return initial_state, initial_exogeneous, initial_action, initial_obs

        def step(self):
            next_state, exogenous = self._create_state(self.idx_counter)
            done = True if self.idx_counter == self.n_market_hours else False
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
        def act(self, state, exogeneous, initial_soc):

            data = {
                "Time": state["time"],
                "Energy demand": state["energy_consumption"],
                "PV production": state["solar_power"],
                "Wholesale prices": exogeneous["wholesale_price"],
                "Retail prices": exogeneous["retail_price"]
            }

            data_time_horizon = pd.DataFrame(data)
            data_time_horizon.reset_index(drop=True, inplace=True)
            df = model.predict(data_time_horizon, initial_soc)

            action = df['Battery_flow'].loc[0]

            return action
        
        def last_act(self, state, exogeneous, initial_soc):
            data = {
                "Time": state["time"],
                "Energy demand": state["energy_consumption"],
                "PV production": state["solar_power"],
                "Wholesale prices": exogeneous["wholesale_price"],
                "Retail prices": exogeneous["retail_price"]
            }

            data_time_horizon = pd.DataFrame(data)
            data_time_horizon.reset_index(drop=True, inplace=True)
            df = model.predict(data_time_horizon, initial_soc)

            action = df['Battery_flow']

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

    time = env.idx_counter
    total_PV_Production = 0
    total_Load = 0
    total_Import = 0
    total_export = 0
    SelfSufficiency = 0
    SelfConsumption = 0


    while done is not True:
        next_state, exogeneous, done = env.step()

        if done:
            action = agent.last_act(next_state, exogeneous, observation['BESS_SOC'])
            for i in range(len(next_state["time"])):
                if i==0:
                    observation = env.new_observation(action[i])
                    print(next_state)
                else:
                    observation = env.last_observation(action[i])

                costs = calculate.obj_function(observation["Trade"], exogeneous["wholesale_price"].loc[i], exogeneous["retail_price"].loc[i])

                df.loc[time, "Time"]=next_state["time"].loc[i]
                df.loc[time, "BESS_SOC"] = observation["BESS_SOC"]
                df.loc[time, "PV Production"] = next_state["solar_power"].loc[i]
                df.loc[time, "Trade"] = observation["Trade"]
                df.loc[time, "Load"] = next_state["energy_consumption"].loc[i]
                df.loc[time, "Battery_flow"]=action[i]
                df.loc[time, "Costs"]=costs
                
                total_PV_Production += next_state["solar_power"].loc[i]
                total_Load += next_state["energy_consumption"].loc[i]
                if observation["Trade"]>=0:
                    total_export+=observation["Trade"]
                else:
                    total_Import+=observation["Trade"]

                time=env.idx_counter
        else:
            action = agent.act(next_state, exogeneous, observation['BESS_SOC'])
            observation = env.new_observation(action)
            costs = calculate.obj_function(observation["Trade"], exogeneous["wholesale_price"].loc[0], exogeneous["retail_price"].loc[0])


            df.loc[time, "Time"]=next_state["time"].loc[0]
            df.loc[time, "BESS_SOC"] = observation["BESS_SOC"]
            df.loc[time, "PV Production"] = next_state["solar_power"].loc[0]
            df.loc[time, "Trade"] = observation["Trade"]
            df.loc[time, "Load"] = next_state["energy_consumption"].loc[0]
            df.loc[time, "Battery_flow"]=action
            df.loc[time, "Costs"]=costs
            
            total_PV_Production += next_state["solar_power"].loc[0]
            total_Load += next_state["energy_consumption"].loc[0]
            if observation["Trade"]>=0:
                total_export+=observation["Trade"]
            else:
                total_Import+=observation["Trade"]

            time=env.idx_counter

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