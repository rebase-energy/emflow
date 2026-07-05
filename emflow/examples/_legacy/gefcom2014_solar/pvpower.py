import pvlib
import numpy as np
import pandas as pd
import emflow as ef

def clearsky_power(pvsystem: ef.PVSystem, datetimes: pd.DatetimeIndex): 

    location = pvsystem.location.to_pvlib()
    clear_sky = location.get_clearsky(datetimes, model='ineichen')
    solar_position = location.get_solarposition(datetimes)
    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        pvsystem.surface_tilt, pvsystem.surface_azimuth,
        solar_position['apparent_zenith'], solar_position['azimuth'],
        dni=clear_sky['dni'], ghi=clear_sky['ghi'], dhi=clear_sky['dhi']
    )
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    cell_temperature = pvlib.temperature.sapm_cell(
        poa_irradiance['poa_global'], 
        temp_air=25, wind_speed=0,  # ambient temperature and wind speed
        **temperature_model_parameters
    )
    # Define system parameters
    module_parameters = {'pdc0': 200, 'gamma_pdc': -0.004}  # pdc0 is the DC power rating at STC
    inverter_parameters = {'pdc0': 200}  # max DC power input for inverter

    # Create a PVSystem and ModelChain object
    pv_system = pvlib.pvsystem.PVSystem(
        surface_tilt=pvsystem.surface_tilt,
        surface_azimuth=pvsystem.surface_azimuth,
        module_parameters=module_parameters,
        inverter_parameters=inverter_parameters
    )

    dc_power = pv_system.pvwatts_dc(poa_irradiance['poa_global'], temp_cell=cell_temperature)
    
    ac_power = pv_system.get_ac("pvwatts", dc_power)

    return ac_power

def physical_power(pvsystem: ef.PVSystem, ghi: pd.DataFrame): 
    datetimes = ghi.index
    location = pvsystem.location.to_pvlib()

    # Calculate solar zenith angle to use in the decomposition
    solar_position = location.get_solarposition(datetimes)
    zenith = solar_position['apparent_zenith']

    # Use the `disc` model to estimate DNI and DHI from GHI
    dni_disc = pvlib.irradiance.disc(ghi, zenith, datetimes)['dni']
    dhi_disc = ghi - dni_disc * np.cos(np.radians(zenith))

    module_parameters = {'pdc0': 1, 'gamma_pdc': -0.004}  # STC power rating and temperature coefficient
    inverter_parameters = {'pdc0': 1}  # Max DC power input for inverter

    # Create PV system and ModelChain
    pv_system = pvlib.pvsystem.PVSystem(
        surface_tilt=pvsystem.surface_tilt,
        surface_azimuth=pvsystem.surface_azimuth,
        module_parameters=module_parameters,
        inverter_parameters=inverter_parameters
    )

    # Calculate plane-of-array (POA) irradiance
    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        pvsystem.surface_tilt,
        pvsystem.surface_azimuth,
        solar_position['apparent_zenith'],
        solar_position['azimuth'],
        dni=dni_disc,
        ghi=ghi,
        dhi=dhi_disc
    )

    # Example ambient temperature and wind speed
    temp_air = 25  # degrees Celsius
    wind_speed = 0  # m/s

    # Calculate cell temperature
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    cell_temperature = pvlib.temperature.sapm_cell(
        poa_irradiance['poa_global'],
        temp_air=temp_air,
        wind_speed=wind_speed,
        **temperature_model_parameters
    )

    # Calculate DC power output
    dc_power = pv_system.pvwatts_dc(poa_irradiance['poa_global'], temp_cell=cell_temperature)

    # Convert DC power to AC power output
    ac_power = pv_system.get_ac("pvwatts", dc_power)

    return ac_power
