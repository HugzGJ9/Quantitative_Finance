import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from API.SUPABASE.client import getDfSupabase
#INFORMATION REGARDING PROFILE DATA:
#Location	Subdivision	            Latitude	Longitude	kWh/day Summer	kWh/day Autumn	kWh/day Winter	kWh/day Spring	Panel Tilt Angle
#Montluçon	Auvergne-Rhône-Alpes	46.3311	    2.5985	        5.82	        3.10	        1.52	        5.09	    40°     South
#Montluçon location Latitude : 46.3311	Longitude : 2.5985 matches with weather data used for forecasts.

SR_LF_Montlucon = {
    "Summer": [0.0, 0.0, 0.0, 0.0, 0.02, 0.10, 0.25, 0.40, 0.50, 0.60, 0.65, 0.67, 0.65, 0.60, 0.55, 0.45, 0.3, 0.18, 0.06, 0.01, 0.0, 0.0, 0.0, 0.0],
    "Autumn": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.14, 0.29, 0.38, 0.45, 0.46, 0.39, 0.38, 0.30, 0.21, 0.08, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Winter": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.10, 0.19, 0.24, 0.26, 0.26, 0.22, 0.16, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Spring": [0.0, 0.0, 0.0, 0.0, 0.01, 0.05, 0.15, 0.30, 0.42, 0.55, 0.61, 0.64, 0.62, 0.59, 0.49, 0.36, 0.22, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
}

SR_LF_Montlucon2 = {
    1: [0, 0, 0, 0, 0, 0, 0, 0, 0.002264, 0.079804, 0.190623, 0.277448, 0.311346, 0.31082, 0.270448, 0.178091, 0.060135, 0.000914, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 0, 0, 0, 0, 0, 0.000154, 0.040149, 0.201589, 0.294156, 0.367536, 0.418661, 0.426892, 0.37059, 0.292155, 0.203316, 0.045935, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 0, 0, 0.00017, 0.025578, 0.156103, 0.29947, 0.404999, 0.483152, 0.533141, 0.536074, 0.4655, 0.378859, 0.279755, 0.144752, 0.014678, 0, 0, 0, 0, 0, 0],
    4: [0, 0, 0, 0, 0.000134, 0.011967, 0.091731, 0.230602, 0.362272, 0.455159, 0.516344, 0.550541, 0.535401, 0.473091, 0.389035, 0.289391, 0.16956, 0.048649, 0.001302, 0, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 0.004031, 0.03872, 0.122628, 0.249058, 0.368193, 0.453598, 0.510078, 0.526858, 0.508251, 0.457951, 0.385895, 0.293672, 0.18166, 0.074686, 0.017711, 0, 0, 0, 0, 0],
    6: [0, 0, 0, 0, 0.009646, 0.046022, 0.130796, 0.260493, 0.382255, 0.474944, 0.51576, 0.542962, 0.536005, 0.48807, 0.420827, 0.321771, 0.20859, 0.093156, 0.032027, 0.002235, 0, 0, 0, 0],
    7: [0, 0, 0, 0, 0.004191, 0.037015, 0.118198, 0.25469, 0.381809, 0.478307, 0.538252, 0.572326, 0.562501, 0.510206, 0.439634, 0.345034, 0.224279, 0.097136, 0.029378, 0.001767, 0, 0, 0, 0],
    8: [0, 0, 0, 0, 0, 0.017516, 0.098485, 0.238383, 0.37605, 0.471353, 0.534377, 0.569277, 0.559448, 0.511013, 0.434951, 0.334816, 0.199689, 0.072456, 0.006316, 0, 0, 0, 0, 0],
    9: [0, 0, 0, 0, 0, 0.002234, 0.072538, 0.216111, 0.353796, 0.453928, 0.518144, 0.540648, 0.523611, 0.455607, 0.368744, 0.265459, 0.130914, 0.015303, 0, 0, 0, 0, 0, 0],
    10: [0, 0, 0, 0, 0, 0, 0.015946, 0.148698, 0.26964, 0.354005, 0.422271, 0.459172, 0.427914, 0.359017, 0.267782, 0.161707, 0.023934, 0, 0, 0, 0, 0, 0, 0],
    11: [0, 0, 0, 0, 0, 0, 0, 0.037312, 0.165269, 0.262424, 0.32517, 0.351797, 0.329664, 0.273437, 0.17597, 0.040093, 0, 0, 0, 0, 0, 0, 0, 0],
    12: [0, 0, 0, 0, 0, 0, 0, 0.001708, 0.072068, 0.1613, 0.258507, 0.30029, 0.289191, 0.214244, 0.112881, 0.027652, 0, 0, 0, 0, 0, 0, 0, 0]
}


def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b
def polyreg(x, a, b, c, d):
    y = a*x**3 + b*x**2 + c*x + d
    return y
def WIND_LoadFactor_FR():
    wind_speed = np.array([
        3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
        8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5,
        13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5,
        18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23
    ])
    generation_pct = np.array([
        1.0, 2.9, 5.3, 8.2, 11.7, 15.9, 21.0, 27.0, 34.0, 41.9,
        51.0, 61.0, 71.9, 83.0, 92.4, 97.6, 99.5, 99.9, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0
    ])

    y_data = generation_pct / 100.0
    p0 = [1, 9, 1, 0]
    params, _ = curve_fit(sigmoid, wind_speed, y_data, p0)
    return params


def SOLAR_LoadFactor_FR(isShow=False):
    weather = getDfSupabase("WeatherFR")
    weather["id"] = pd.to_datetime(weather["id"], utc=True)
    weather = (weather
               .set_index("id")
               .sort_index()
               .loc[:, ["Solar_Radiation"]])
    df = weather.groupby([weather.index.month, weather.index.hour]).mean()
    df.index.set_names(['month', 'hour'], inplace=True)
    f_reset = df.reset_index()
    # Step 3: Melt SR_LoadFactor_FR2 to long format
    load_factor_long = SR_LoadFactor_FR2.drop(columns="Hour").reset_index().melt(
        id_vars='index',
        var_name='month',
        value_name='Load_Factor'
    )
    load_factor_long.rename(columns={'index': 'hour'}, inplace=True)
    # Step 4: Ensure types match
    load_factor_long['month'] = load_factor_long['month'].astype(int)
    load_factor_long['hour'] = load_factor_long['hour'].astype(int)
    # Step 5: Merge
    merged = pd.merge(f_reset, load_factor_long, how='left', on=['month', 'hour'])
    # Step 6: (Optional) Set back to MultiIndex
    merged.set_index(['month', 'hour'], inplace=True)
    params = np.polyfit(merged['Solar_Radiation'], merged['Load_Factor'], 3)
    if isShow:
        plt.scatter(merged['Solar_Radiation'], merged['Load_Factor'])
        y_hat = np.poly1d(params)(merged['Solar_Radiation'])
        plt.plot(merged['Solar_Radiation'], y_hat, "r--", lw=1)
        plt.show()
    return params
# Create final dataframe
SR_LoadFactor_FR = pd.DataFrame({
    "Hour": [f"{h:02}:00" for h in range(24)],
    "Summer": SR_LF_Montlucon["Summer"],
    "Autumn": SR_LF_Montlucon["Autumn"],
    "Winter": SR_LF_Montlucon["Winter"],
    "Spring": SR_LF_Montlucon["Spring"]
})

SR_LoadFactor_FR2 = pd.DataFrame(SR_LF_Montlucon2)
SR_LoadFactor_FR2['Hour'] = [f"{h:02}:00" for h in range(24)]

if __name__ == '__main__':
    WIND_LoadFactor_FR()