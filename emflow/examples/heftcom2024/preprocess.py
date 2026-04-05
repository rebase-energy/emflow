import numpy as np
import pandas as pd
import xarray as xr

dwd_Hornsea1 = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_features = dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")

dwd_solar = xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
dwd_solar_features = dwd_solar["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features["ref_datetime"] = dwd_solar_features["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")

energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]

data = dwd_Hornsea1_features.merge(dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
data = data.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
data = data.drop(columns="ref_datetime",axis=1).reset_index()
data = data.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")
data = data[data["valid_datetime"] - data["ref_datetime"] < np.timedelta64(50,"h")]
data.rename(columns={"WindSpeed:100":"WindSpeed"}, inplace=True)
data["total_generation_MWh"] = data["Wind_MWh_credit"] + data["Solar_MWh_credit"]
data.drop(columns=["dtm"], inplace=True)

# DWD coordinates for Hornsea1
dwd_coords_Hornsea1 = pd.DataFrame({'longitude': dwd_Hornsea1.longitude.values, 'latitude': dwd_Hornsea1.latitude.values, })
dwd_coords_Hornsea1.to_csv('data/dwd_coords_Hornsea1.csv', index=False)

# DWD coordinates for PES10
dwd_coords_pes10 = pd.DataFrame({'longitude': dwd_solar.longitude.values, 'latitude': dwd_solar.latitude.values, })
dwd_coords_pes10.to_csv('data/dwd_coords_pes10.csv', index=False)