import os
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import emflow as ef

import matplotlib.pyplot as plt

# Get the parent directory (one level up)
parent_dir = Path.cwd().parent
df_data = pd.read_csv(os.path.join(parent_dir, 'data', 'gefcom2014', 'gefcom2014-solar.csv'),
                index_col=[0, 1], parse_dates=True, header=[0, 1])
df_scores = pd.read_csv(os.path.join(parent_dir, 'data', 'gefcom2014', 'gefcom2014-solar-scores.csv'),
                        index_col=0)

pvsystem_1 = ef.PVSystem(name="Site1",
                         capacity=1,
                         longitude=145,
                         latitude=-37.5,
                         surface_azimuth=38,
                         surface_tilt=36)

pvsystem_2 = ef.PVSystem(name="Site2",
                         capacity=1,
                         longitude=145,
                         latitude=-37.5,
                         surface_azimuth=327,
                         surface_tilt=35)

pvsystem_3 = ef.PVSystem(name="Site3",
                         capacity=1,
                         longitude=145,
                         latitude=-37.5,
                         surface_azimuth=31,
                         surface_tilt=21)

portfolio = ef.Portfolio(name="Portfolio", assets=[pvsystem_1, pvsystem_2, pvsystem_3])

dataset = ef.Dataset(name="gefcom2024-solar",
                     description="Data provided by the organisers of HEFTCom2024. Participants are free to use additional external data.",
                     collection=portfolio,
                     data={"data_gefcom2014_solar": df_data, "scores_gefcom2014_solar": df_scores})

state_space = ef.DataFrameSpace({asset.name: {
    'U10': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    'V10': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    'U100': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    'V100': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    'Power': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
} for asset in portfolio.assets})

n_quantiles = 99

action_space = ef.DataFrameSpace({asset.name: {
    f"Quantile_forecast_{i+1}": gym.spaces.Box(low=0, high=1, shape=(1,)) for i in range(n_quantiles)
} for asset in portfolio.assets})


class GEFCom2014SolarEnv(gym.Env):
    def __init__(self, dataset: ef.Dataset): 
        self.dataset = dataset
        self.data = dataset.data["data_gefcom2014_solar"]
        self.scores = dataset.data["scores_gefcom2014_solar"]
        self.input = ['VAR134', 'VAR157', 'VAR164', 'VAR165', 'VAR166', 'VAR167', 'VAR169', 'VAR175', 'VAR178', 'VAR228', 'VAR78', 'VAR79']
        self.target = ["Power"]
        self.state_space = state_space
        self.action_space = action_space
        self.idx_counter = 0

        self.train = [["2012-04-01 01:00:00", "2013-04-01 00:00:00"],
                ["2013-04-01 01:00:00", "2013-05-01 00:00:00"],
                ["2013-05-01 01:00:00", "2013-06-01 00:00:00"],
                ["2013-06-01 01:00:00", "2013-07-01 00:00:00"],
                ["2013-07-01 01:00:00", "2013-08-01 00:00:00"],
                ["2013-08-01 01:00:00", "2013-09-01 00:00:00"],
                ["2013-09-01 01:00:00", "2013-10-01 00:00:00"],
                ["2013-10-01 01:00:00", "2013-11-01 00:00:00"],
                ["2013-11-01 01:00:00", "2013-12-01 00:00:00"],
                ["2013-12-01 01:00:00", "2014-01-01 00:00:00"],
                ["2014-01-01 01:00:00", "2014-02-01 00:00:00"],
                ["2014-02-01 01:00:00", "2014-03-01 00:00:00"],
                ["2014-03-01 01:00:00", "2014-04-01 00:00:00"],
                ["2014-04-01 01:00:00", "2014-05-01 00:00:00"],
                ["2014-05-01 01:00:00", "2014-06-01 00:00:00"]]
        self.test = [["2013-04-01 01:00:00", "2013-05-01 00:00:00"],
                ["2013-05-01 01:00:00", "2013-06-01 00:00:00"],
                ["2013-06-01 01:00:00", "2013-07-01 00:00:00"],
                ["2013-07-01 01:00:00", "2013-08-01 00:00:00"],
                ["2013-08-01 01:00:00", "2013-09-01 00:00:00"],
                ["2013-09-01 01:00:00", "2013-10-01 00:00:00"],
                ["2013-10-01 01:00:00", "2013-11-01 00:00:00"],
                ["2013-11-01 01:00:00", "2013-12-01 00:00:00"],
                ["2013-12-01 01:00:00", "2014-01-01 00:00:00"],
                ["2014-01-01 01:00:00", "2014-02-01 00:00:00"],
                ["2014-02-01 01:00:00", "2014-03-01 00:00:00"],
                ["2014-03-01 01:00:00", "2014-04-01 00:00:00"],
                ["2014-04-01 01:00:00", "2014-05-01 00:00:00"],
                ["2014-05-01 01:00:00", "2014-06-01 00:00:00"],
                ["2014-06-01 01:00:00", "2014-07-01 00:00:00"]]

        self.n_steps = len(self.test)

    def reset(self):
        self.idx_counter = 0
        initial_dataframe = self.data.loc[(self.data.index.get_level_values('valid_datetime') >= self.train[self.idx_counter][0]) &
                                          (self.data.index.get_level_values('valid_datetime') <= self.train[self.idx_counter][1])]

        initial_input = initial_dataframe.loc[:,(slice(None), env.input)]
        initial_target = initial_dataframe.loc[:,(slice(None), env.target)]
        initial_data = {"input": initial_input, "target": initial_target}

        first_input = self.data.loc[(self.data.index.get_level_values('valid_datetime') >= self.test[self.idx_counter][0]) &
                                    (self.data.index.get_level_values('valid_datetime') <= self.test[self.idx_counter][1]),
                                     pd.IndexSlice[:, self.input]]

        return initial_data, first_input

    def step(self, action=None):

        if self.idx_counter+1 < self.n_steps:
            next_input = self.data.loc[(self.data.index.get_level_values('valid_datetime') >= self.test[self.idx_counter+1][0]) &
                                       (self.data.index.get_level_values('valid_datetime') <= self.test[self.idx_counter+1][1]),
                                         pd.IndexSlice[:, self.input]] 

            next_target = self.data.loc[(self.data.index.get_level_values('valid_datetime') >= self.train[self.idx_counter+1][0]) &
                                 (self.data.index.get_level_values('valid_datetime') <= self.train[self.idx_counter+1][1]),
                                  pd.IndexSlice[:, self.target]]

            done = False
            
            self.idx_counter += 1

            return next_input, next_target, done
    
        elif self.idx_counter+1 == self.n_steps:
            next_target = self.data.loc[(self.data.index.get_level_values('valid_datetime') >= self.test[self.idx_counter][0]) &
                                        (self.data.index.get_level_values('valid_datetime') <= self.test[self.idx_counter][1]),
                                  pd.IndexSlice[:, self.target]]
            
            done = True

            self.idx_counter += 1
            
            return None, next_target, done


    def plot_overall_results(self, losses, drop_tasks=None, n_top_teams=None, xlim=None):
        df_scores = self.scores
        df_scores = df_scores.assign(**losses)
        df_scores = df_scores.drop(index=drop_tasks)
        df_scores = df_scores.mean()
        df_scores = df_scores.sort_values()

        colors = [(0.66, 0.66, 0.66, 0.7)] * len(df_scores)
        n = len(losses)
        blues = [plt.cm.Blues(i / (n + 1)) for i in range(1, n + 1)]

        for key_idx, key in enumerate(losses.keys()):
            idx = list(df_scores.index).index(key)
            colors[idx] = blues[key_idx]

        ax = df_scores.plot.bar(color=colors)
        ax.set_ylabel("Pinball loss")
        if xlim: ax.set_xlim(0, xlim)
        
        return ax

    def plot_results(self, losses, drop_tasks=None, n_top_teams=None, xlim=None):
        df_scores = self.scores
        df_scores = df_scores.assign(**losses)
        df_scores = df_scores.drop(index=drop_tasks)
        df_scores.loc["Overall"] = df_scores.mean()
        teams = list(df_scores.drop(columns=losses.keys()).loc["Overall",:].sort_values().iloc[:n_top_teams].index.values)
        teams.extend(losses.keys())
        ax = df_scores.loc[::-1,teams].plot.barh(title="Pinball loss GEFCom20214")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if xlim: ax.set_xlim(0, xlim)

        return ax

    def plot_forecasts(self, training_target, df_predictions, site="Site1"): 
        import pandas as pd
        import plotly.graph_objects as go
        df = df_predictions.droplevel(0).loc[:,"Site1"]
        df_target = training_target.loc[df.index[0]:df.index[-1],:].droplevel(0).loc[:,"Site1"]

        # Sort quantiles to ensure outer quantiles are filled first
        quantile_columns = [col for col in df.columns if col.startswith('quantile')]
        quantile_columns.sort(key=lambda x: float(x.split('_')[1]))

        # Create the figure
        fig = go.Figure()

        # Loop over the quantiles and add shaded areas
        # Start from the outermost quantiles and move inward
        for i in range(len(quantile_columns) // 2):
            upper_quantile = quantile_columns[-(i+1)]
            lower_quantile = quantile_columns[i]
            
            # Add the upper quantile line (invisible)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[upper_quantile],
                mode='lines',
                line=dict(color='rgb(0, 0, 255)', width=0),  # No line for the upper bound
                fill=None,
                showlegend=False,
                hoverinfo='skip',  # This disables the hover tooltip
                name=f'{upper_quantile}'
            ))
            
            # Add the lower quantile line and fill the area between the two traces
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[lower_quantile],
                mode='lines',
                line=dict(width=0),  # No line for the lower bound
                fill='tonexty',  # Fill between this trace and the previous one
                fillcolor=f'rgba(0, 0, 255, {0.1 + i * 0.1})',  # Increasing opacity for each quantile band
                showlegend=False,
                hoverinfo='skip',  # This disables the hover tooltip
                name=f'{lower_quantile}'
            ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["quantile_50"],
            mode='lines',
            line=dict(color='blue', width=1),  # Central forecast line
            showlegend=False,
            name='Median'
        ))

        fig.add_trace(go.Scatter(
            x=df_target.index,
            y=df_target["Power"],
            mode='lines',
            line=dict(color='rgba(200, 0, 0, 0.5)', width=2),  # Central forecast line
            showlegend=False,
            name='Power'
        ))

        # Customize layout
        fig.update_layout(
            title='GEFCom2014 Wind Power Forecast',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode="x"
        )

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

        # Show the plot
        fig.show()


env = GEFCom2014SolarEnv(dataset=dataset)


from emflow.problems.objective import PinballLoss

obj = PinballLoss(quantiles=np.arange(0.01, 1, 0.01))

from emflow import Problem

problem = Problem(name="gefcom2014-solar", 
                  environment=env,
                  objective=obj)
