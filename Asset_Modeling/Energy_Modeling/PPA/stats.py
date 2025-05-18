import random

import pandas as pd
from matplotlib import pyplot as plt
from Model.Power.dataProcessing import plot_hexbin_density
from API.SUPABASE.client import getDfSupabase

import numpy as np
import pandas as pd
import scipy.stats as st
from Logger.Logger import mylogger

def pltSolarGen():
    rad = np.linspace(0, 900, 300)
    lf = simulate_solar_generation(rad, 1.0)
    plt.plot(rad, lf)
    plt.xlabel("Solar Radiation (W/m²)")
    plt.ylabel("Load Factor")
    plt.title("Solar Generation Curve")
    plt.grid(True)
    plt.show()
def pltWindGen():
    wind = np.linspace(0, 30, 100)
    lf = simulate_wind_generation(wind, 1.0)
    plt.plot(wind, lf)
    plt.xlabel("Wind Speed")
    plt.ylabel("Load Factor")
    plt.title("Wind Generation Curve")
    plt.grid(True)
    plt.show()

def simulate_solar_generation(radiation, capacity):
    from Asset_Modeling.Energy_Modeling.PPA.Profile_modelisation import SOLAR_LoadFactor_FR

    is_scalar = np.isscalar(radiation)
    radiation = np.asarray(radiation)

    params = SOLAR_LoadFactor_FR()
    generation = np.poly1d(params)(radiation)
    generation = np.clip(generation, 0, 1)
    generation = generation * capacity
    return float(generation) if is_scalar else generation

def simulate_wind_generation(wind_speed, capacity):
    from Asset_Modeling.Energy_Modeling.PPA.Profile_modelisation import WIND_LoadFactor_FR, sigmoid
    wind_speed = wind_speed / 3.6 #convert km/hour to m/s
    is_scalar = np.isscalar(wind_speed)
    wind_speed = np.asarray(wind_speed)
    params = WIND_LoadFactor_FR()
    generation = sigmoid(wind_speed, *params)
    generation = np.clip(generation, 0, 1)
    generation = generation * capacity
    return float(generation) if is_scalar else generation

def buildSyntheticGenerationSR(df: pd.DataFrame,
                               capacity: float = 10.0) -> pd.DataFrame:
    out = df.copy()
    out['generation'] = 0.0
    generation = simulate_solar_generation(out['Solar_Radiation'],
                                           capacity=capacity
                                           )
    out['generation'] = generation
    return out

def buildSyntheticGenerationWIND(df: pd.DataFrame,
                               capacity: float = 10.0) -> pd.DataFrame:
    out = df.copy()
    out['generation'] = 0.0

    generation = simulate_wind_generation(out['Wind_Speed_100m'],
                                           capacity=capacity,
                                           )
    out['generation'] = generation
    return out

if __name__ == '__main__':
    pltSolarGen()
    print('test')