import matplotlib.pyplot as plt
import pandas as pd

def timeseries(
    start_date = '2015-01-01',
    end_date = '2016-01-01',
    df = None,
    plot_columns = None      
):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_data = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]

    # Plotting
    plt.figure(figsize=(12, 6))
    for idx, name in enumerate(plot_columns):
        plt.plot(filtered_data['Time'], filtered_data[name], label=name)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Energy [kWh]', fontsize=16)
    plt.title(f'PV-BESS system operation between {start_date} and {end_date}', fontsize=18)
    plt.grid(True)
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()