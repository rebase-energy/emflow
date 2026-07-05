import pandas as pd
import os
import numpy as np

def import_load_from_CSV(selected_year, building_number):
    # script_dir = os.path.dirname(__file__)
    # parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    data_folder = 'Data\Loads'
    file_name = f"VillaElectricity-{selected_year}.csv"

    #file_path = os.path.join(parent_dir, data_folder, file_name)
    file_path = os.path.join(data_folder, file_name)

    df = pd.read_csv(file_path)

    if selected_year >= 2020:
        building_name = f"brf{building_number}"
    else:
        building_name = f"villa{building_number}"

    if building_name in df.columns:
        energy_consumption = df[building_name]
    else:
        raise ValueError("Building not found")
    
    return energy_consumption


def import_from_Excel(excel_path, sheet_name, column_name):
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if column_name in df.columns:
            column_data = df[column_name]
    else:
        raise ValueError("Column not found")
    
    return column_data


def calculate_retail_price(wholesale_price):
     network_fee = np.ones_like(wholesale_price)*0.317
     electricity_tax = np.ones_like(wholesale_price)*0.428
     electric_fee = np.ones_like(wholesale_price)*0.05
     Elcertifikat = np.ones_like(wholesale_price)*0
     VAT = np.ones_like(wholesale_price)*0.25

     retail_price = (wholesale_price + network_fee + electricity_tax + electric_fee + Elcertifikat)*(1+VAT)
     return retail_price


def find_load(selected_year, annual_load_req, boundaries=0.1):
    building_list = []
    loads_list = []

    for i in range(1, 109):
        load = import_load_from_CSV(selected_year, i)
        annual_load = sum(load)

        if annual_load >= (1-boundaries)*annual_load_req and annual_load <= (1+ boundaries)*annual_load_req:
            building_list.append(i)
            loads_list.append(annual_load)

    return building_list, loads_list