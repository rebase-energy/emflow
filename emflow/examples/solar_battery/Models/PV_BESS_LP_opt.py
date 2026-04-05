import gymnasium as gym
import emflow as ef
import energydatamodel as edm
from emflow.problems.dataset import Dataset
from emflow.problems.objective import Objective

import pandas as pd
import datetime
import os
import timeit
from ortools.linear_solver import pywraplp


import energydatamodel as edm
import numpy as np
import matplotlib.pyplot as plt


def PV_BESS_LP_solver(
        database,
        EnergySystem,
        initial_soc=0.1,
        trade_limit=6,
        export_solutions_to_Excel=False
):
    PVsyst = EnergySystem.assets[0]
    BESSsyst = EnergySystem.assets[1]
    
    # Load timeseries data
    market_df = database.iloc[:,:5]
    market_df.columns = ["time", "load", "solar", "wholesale_price", "retail_price"]
    market_df = market_df[~pd.isnull(market_df["time"])].fillna(0)
    market1_df = market_df.copy()
    market1_df.sort_values(by=["time"], inplace=True)
    market1_df["time_string"] = market1_df.apply(lambda x:(x["time"]+ datetime.timedelta(seconds=0.002)).strftime("%d/%m/%Y %H:%M"), axis=1)
    market1_df.set_index("time_string", inplace=True)
    market_df = market1_df

    column_names = ["max_import_power", "max_export_power"]
    row_data = [trade_limit, trade_limit] 
    grid_df = pd.DataFrame([row_data], columns=column_names)

    column_names = ["max_charge_rate", "max_discharge_rate", "charge_eff", "discharge_eff", "min_soc", "max_soc", "initial_soc"]
    row_data = [BESSsyst.max_charge, BESSsyst.max_discharge, BESSsyst.charge_efficiency, BESSsyst.discharge_efficiency, BESSsyst.min_soc, 1, initial_soc]
    batt_df = pd.DataFrame([row_data], columns=column_names)


    # Convert dataframe to dictionary
    marketDict = market_df.to_dict()
    gridDict = grid_df.to_dict()
    battDict = batt_df.to_dict()

    timeInterval = market_df.iloc[1]["time"] - market_df.iloc[0]["time"]

    # Assign the data to right places
    input = type("input", (dict,), {})()
    input.update({
        "simData": {
            "startTime": datetime.datetime.strptime(market_df.index[0], "%d/%m/%Y %H:%M"),
            "dt": int(round(timeInterval.total_seconds())) / (60 * 60), #in hour
            "tIndex": market_df.shape[0]
            },
        "market": {
            key: {
                sub_key: sub_item for sub_key, sub_item in marketDict[key].items()
                } for key in marketDict.keys() if key != "time"
            },
        "grid": {
            key: item[0] for key, item in gridDict.items()
            },
        "batt": {
            key: item[0] for key, item in battDict.items()
            }
        })

    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver.CreateSolver("CBC")

    inf = solver.infinity()

    tIndex = input["simData"]["tIndex"] # number of timeslots
    dt = input["simData"]["dt"]         # time interval in hour

    # Create datetime array
    timestamp=database['Time'].dt.tz_localize(None)
    time = [timestamp[i].strftime("%d/%m/%Y %H:%M") for i in range(len(timestamp))]

    time_s = timeit.default_timer()

    # Add timeseries variables
    vCharge = [solver.NumVar(lb=0, ub=inf, name="") for _ in range(tIndex)]
    vDischarge = [solver.NumVar(lb=-inf, ub=0, name="") for _ in range(tIndex)]
    vSOC = [solver.NumVar(lb=0, ub=inf, name="") for _ in range(tIndex)]
    vImport = [solver.NumVar(lb=-inf, ub=0, name="") for _ in range(tIndex)]
    vExport = [solver.NumVar(lb=0, ub=inf, name="") for _ in range(tIndex)]

    # Add constraints
    for i in range(tIndex):
        
        t = time[i]
        
        # Grid constraints
        solver.Add(vImport[i] + vExport[i] == -input["market"]["load"][t] + input["market"]["solar"][t] - (vDischarge[i] + vCharge[i])) # Eqn. 1
        solver.Add(vExport[i] <= input["grid"]["max_export_power"])    # Eqn. 2(a)
        solver.Add(vImport[i] >= - input["grid"]["max_import_power"])  # Eqn. 2(b)

        # Battery constraints
        solver.Add(vDischarge[i] + vCharge[i] <= input["batt"]["max_charge_rate"] * BESSsyst.storage_capacity)      # Eqn. 3(a)
        solver.Add(vDischarge[i] + vCharge[i] >= -input["batt"]["max_discharge_rate"] * BESSsyst.storage_capacity)  # Eqn. 3(b)
        
        if i == 0:
            solver.Add(vSOC[i] == input["batt"]["initial_soc"]*BESSsyst.storage_capacity + dt * (vCharge[i] * (input["batt"]["charge_eff"]) + vDischarge[i] / (input["batt"]["discharge_eff"]))) # Eqn. 4
        else:
            solver.Add(vSOC[i] == vSOC[i-1] + dt * (vCharge[i] * (input["batt"]["charge_eff"]) + vDischarge[i] / (input["batt"]["discharge_eff"]))) # Eqn. 4
            
        solver.Add(vSOC[i] >= input["batt"]["min_soc"]*BESSsyst.storage_capacity) # Eqn. 5
        solver.Add(vSOC[i] <= input["batt"]["max_soc"]*BESSsyst.storage_capacity) # Eqn. 5

    # Add objective
    obj = 0
    obj += sum([-vExport[i]*input["market"]["wholesale_price"][time[i]] -vImport[i]*input["market"]["retail_price"][time[i]] for i in range(tIndex)])
    solver.Minimize(obj)

    status = solver.Solve()


    time_e = timeit.default_timer()
    runTime = round(time_e - time_s, 4)

    if status == solver.OPTIMAL or status == solver.FEASIBLE:
        #print("Solution is found.")
        #print("Number of variables =", solver.NumVariables())
        #print("Number of constraints =", solver.NumConstraints())
        #print("Computation time = ", runTime)
        objValue = round(solver.Objective().Value(), 2)
        objValueDF = pd.DataFrame.from_dict({"obj_value": objValue}, orient="index", columns=["Total Cost of Importing Power (kr)"])


        result = list(zip([round(vCharge[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vDischarge[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vSOC[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vImport[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vExport[i].solution_value(), 4) for i in range(tIndex)]
                        ))
        result_df = pd.DataFrame(result, index=timestamp, columns=["Charging Power (kW)", "Discharging Power (kW)", "State-of-charge (SOC)", "Import (kW)", "Export (kW)"])
        
        if export_solutions_to_Excel:
            # Extract solution values
            output_folder = "output"
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            excelWriter = pd.ExcelWriter('output/Result.xlsx')
            objValueDF.to_excel(excelWriter, sheet_name='Cost')
            result_df.to_excel(excelWriter, sheet_name='Operation')
            excelWriter.close()

    else:
        print("Solution cannot be found.")

    trade = result_df['Export (kW)'] + result_df['Import (kW)']
    battery_flow = result_df['Charging Power (kW)'] + result_df['Discharging Power (kW)']

    data = {
        'Time': database['Time'], 
        'BESS_SOC': result_df['State-of-charge (SOC)'].reset_index(drop=True),
        'PVProduction': database['PVProduction'],
        'Trade': trade.reset_index(drop=True),
        'Load': database['EnergyDemand'],
        'BatteryFlow': battery_flow.reset_index(drop=True)
    }
    df = pd.DataFrame(data)

    return df, objValue















def PV_BESS_LP_solver_OLD(
        database,
        EnergySystem,
        PVsyst,
        BESSsyst,
        initial_soc=0.1,
        Export_solutions_to_Excel=False
):

    # Load timeseries data
    marketDF = database.iloc[:,:5]
    marketDF.columns = ["time", "load", "solar", "wholesale_price", "retail_price"]
    marketDF = marketDF[~pd.isnull(marketDF["time"])].fillna(0)

    market1DF = marketDF.copy()
    market1DF.sort_values(by=["time"], inplace=True)
    market1DF["time_string"] = market1DF.apply(lambda x:(x["time"]+ datetime.timedelta(seconds=0.002)).strftime("%d/%m/%Y %H:%M"), axis=1)
    market1DF.set_index("time_string", inplace=True)
    marketDF = market1DF

    column_names = ["max_buy_power", "max_sell_power", "max_import_power", "max_export_power"]
    row_data = [6, 6, 6, 6] 
    gridDF = pd.DataFrame([row_data], columns=column_names)

    column_names = ["max_charge_rate", "max_discharge_rate", "capacity", "charge_eff", "discharge_eff", "min_soc", "max_soc", "initial_soc"]
    row_data = [BESSsyst.max_charge, BESSsyst.max_discharge, BESSsyst.storage_capacity, BESSsyst.charge_efficiency, BESSsyst.discharge_efficiency, BESSsyst.min_soc, 1, initial_soc]
    battDF = pd.DataFrame([row_data], columns=column_names)


    # Convert dataframe to dictionary
    marketDict = marketDF.to_dict()
    gridDict = gridDF.to_dict()
    battDict = battDF.to_dict()

    timeInterval = marketDF.iloc[1]["time"] - marketDF.iloc[0]["time"]

    # Assign the data to right places
    input = type("input", (dict,), {})()
    input.update({
        "simData": {
            "startTime": datetime.datetime.strptime(marketDF.index[0], "%d/%m/%Y %H:%M"),
            "dt": int(round(timeInterval.total_seconds())) / (60 * 60), #in hour
            "tIndex": marketDF.shape[0]
            },
        "market": {
            key: {
                sub_key: sub_item for sub_key, sub_item in marketDict[key].items()
                } for key in marketDict.keys() if key != "time"
            },
        "grid": {
            key: item[0] for key, item in gridDict.items()
            },
        "batt": {
            key: item[0] for key, item in battDict.items()
            }
        })

    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver.CreateSolver("CBC")

    inf = solver.infinity()

    tIndex = input["simData"]["tIndex"] # number of timeslots
    dt = input["simData"]["dt"] # time interval in hour

    # Create datetime array
    timestamp=database['Time'].dt.tz_localize(None)
    time = [timestamp[i].strftime("%d/%m/%Y %H:%M") for i in range(len(timestamp))]

    time_s = timeit.default_timer()
    # Add timeseries variables
    vGrid = [solver.NumVar(lb=-inf, ub=inf, name="") for _ in range(tIndex)]
    vBattPower = [solver.NumVar(lb=-inf, ub=inf, name="") for _ in range(tIndex)]
    vCharge = [solver.NumVar(lb=0, ub=inf, name="") for _ in range(tIndex)]
    vDischarge = [solver.NumVar(lb=-inf, ub=0, name="") for _ in range(tIndex)]
    vChargeStatus = [solver.BoolVar(name="") for _ in range(tIndex)]
    vSOC = [solver.NumVar(lb=0, ub=1, name="") for _ in range(tIndex)]
    vTradeStatus = [solver.BoolVar(name="") for _ in range(tIndex)]
    vImport = [solver.NumVar(lb=-inf, ub=0, name="") for _ in range(tIndex)]
    vExport = [solver.NumVar(lb=0, ub=inf, name="") for _ in range(tIndex)]

# Add constraints
    for i in range(tIndex):
        
        t = time[i]
        
        # Grid constraints
        solver.Add(vGrid[i] == -input["market"]["load"][t] + input["market"]["solar"][t] - vBattPower[i]) # Eqn. 1
        solver.Add(vGrid[i] <= input["grid"]["max_sell_power"]) # Eqn. 2
        solver.Add(vGrid[i] >= -input["grid"]["max_buy_power"]) # Eqn. 2
        solver.Add(input["market"]["load"][t] - input["market"]["solar"][t] + (vDischarge[i] + vCharge[i]) <= input["grid"]["max_import_power"]) # Eqn. 3
        solver.Add(input["market"]["load"][t] - input["market"]["solar"][t] + (vDischarge[i] + vCharge[i]) >= -input["grid"]["max_export_power"]) # Eqn. 3

        solver.Add(vGrid[i] == vImport[i] + vExport[i])
        solver.Add(vExport[i] <= input["grid"]["max_export_power"] * vTradeStatus[i])
        solver.Add(vImport[i] >= - input["grid"]["max_import_power"] * (1 - vTradeStatus[i]))

        # Battery constraints
        solver.Add(vBattPower[i] == vCharge[i] + vDischarge[i]) # Eqn. 4
        solver.Add(vCharge[i] <= input["batt"]["max_charge_rate"] *BESSsyst.storage_capacity * vChargeStatus[i]) # Eqn. 5(a)
        solver.Add(vDischarge[i] >= -input["batt"]["max_discharge_rate"] *BESSsyst.storage_capacity * (1-vChargeStatus[i])) # Eqn. 5(b)
        
        if i == 0:
            solver.Add(vSOC[i] == input["batt"]["initial_soc"] + dt / input["batt"]["capacity"] * (vCharge[i] * (input["batt"]["charge_eff"]) + vDischarge[i] / (input["batt"]["discharge_eff"]))) # Eqn. 6
        else:
            solver.Add(vSOC[i] == vSOC[i-1] + dt / input["batt"]["capacity"] * (vCharge[i] * (input["batt"]["charge_eff"]) + vDischarge[i] / (input["batt"]["discharge_eff"]))) # Eqn. 6
            
        solver.Add(vSOC[i] >= input["batt"]["min_soc"]) # Eqn. 7
        solver.Add(vSOC[i] <= input["batt"]["max_soc"]) # Eqn. 7

    # Add objective
    obj = 0
    obj += sum([-vExport[i]*input["market"]["wholesale_price"][time[i]] -vImport[i]*input["market"]["retail_price"][time[i]] for i in range(tIndex)])
    solver.Minimize(obj)

    status = solver.Solve()


    time_e = timeit.default_timer()
    runTime = round(time_e - time_s, 4)

    if status == solver.OPTIMAL or status == solver.FEASIBLE:
        # print("Solution is found.")
        # print("Number of variables =", solver.NumVariables())
        # print("Number of constraints =", solver.NumConstraints())
        # print("Computation time = ", runTime)
        objValue = round(solver.Objective().Value(), 2)
        # print("Objective function = ", objValue)
        
        
        
        objValueDF = pd.DataFrame.from_dict({"obj_value": objValue}, orient="index", columns=["Total Cost of Importing Power (kr)"])
        
        result = list(zip([round(vGrid[i].solution_value(), 4) for i in range(tIndex)], 
                        [round(vBattPower[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vCharge[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vDischarge[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vSOC[i].solution_value(), 4) for i in range(tIndex)],
                        [int(vChargeStatus[i].solution_value()) for i in range(tIndex)],
                        [int(vTradeStatus[i].solution_value()) for i in range(tIndex)],
                        [round(vImport[i].solution_value(), 4) for i in range(tIndex)],
                        [round(vExport[i].solution_value(), 4) for i in range(tIndex)]
                        ))
        resultDF = pd.DataFrame(result, index=timestamp, columns=["Grid Power Flow (kW)", "Battery Output (kW)", "Charging Power (kW)", "Discharging Power (kW)", "State-of-charge (SOC)", "Charge Status", "Trade Status", "Import (kW)", "Export (kW)"])
        
        if Export_solutions_to_Excel:
            # Extract solution values
            output_folder = "output"
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            excelWriter = pd.ExcelWriter('output/Result.xlsx')
            objValueDF.to_excel(excelWriter, sheet_name='Cost')
            resultDF.to_excel(excelWriter, sheet_name='Operation')
            excelWriter.close()

    else:
        print("Solution cannot be found.")

    data = {
        'Time': database['Time'].dt.tz_localize(None),
        'BESS_SOC': resultDF['State-of-charge (SOC)'].reset_index(drop=True),
        'PV Production': marketDF['solar'].reset_index(drop=True),
        'Trade': resultDF['Grid Power Flow (kW)'].reset_index(drop=True),
        'Load': marketDF['load'].reset_index(drop=True),
        'Battery_flow': resultDF['Battery Output (kW)'].reset_index(drop=True),
        'Battery_charge': resultDF['Charging Power (kW)'].reset_index(drop=True),
        'Battery_discharge': resultDF['Discharging Power (kW)'].reset_index(drop=True),
        'BESS_status': resultDF['Charge Status'].reset_index(drop=True)
    }
    df = pd.DataFrame(data)

    return df, objValue



def PV_BESS_LP(
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


    class Predictor(ef.Predictor):
        def predict(self, database, EnergySystem, PVsyst, BESSsyst):
            df, objValue = PV_BESS_LP_solver(database, 
                                EnergySystem, 
                                PVsyst, 
                                BESSsyst)
            
            battery_flow = df["Battery_flow"]

            return battery_flow, objValue
        
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

            prediction, objective_function = model.predict(database, EnergySystem, PVsyst, BESSsyst)

            return initial_state, initial_exogeneous, initial_action, initial_obs, prediction, objective_function

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

        def act(self, state, obs, exogeneous, time, prediction):
            action = prediction[time]
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
    initial_state, initial_exogeneous, initial_action, observation, prediction, objective_function = env.reset()
    
    time = 0
    total_PV_Production = 0
    total_Load = 0
    total_Import = 0
    total_export = 0
    SelfSufficiency = 0
    SelfConsumption = 0

    while done is not True:
        next_state, exogeneous, done = env.step()
        action = agent.act(next_state, observation, exogeneous, time, prediction)
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


    return df, objective_function, KPI
