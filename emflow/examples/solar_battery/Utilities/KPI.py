import pandas as pd


def calculate(
        df
):
        total_PV_production = sum(df['PVProduction'])
        total_load = sum(df['Load'])
        total_export = sum(df['Trade'][(df['Trade']>0)])
        total_import = sum(df['Trade'][(df['Trade']<0)])

        solar_fraction = total_PV_production/total_load
        self_consumption = (total_PV_production-total_export)/total_PV_production
        self_sufficiency = (total_PV_production-total_export)/total_load

        data = {
        'SolarFraction': solar_fraction,
        'SelfConsumption': self_consumption,
        'SelfSufficiency': self_sufficiency
        }
        kpi = pd.DataFrame(data, index=[0])

        return kpi