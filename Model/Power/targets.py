from __future__ import annotations

from typing import Dict, List

TARGETS: Dict[str, List[str]] = {
    "WIND": ['Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
             'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance', 'Cloud_Cover', 'Cloud_Cover_Low',
             'Cloud_Cover_Mid', 'Cloud_Cover_High', 'Temperature_2m', 'Relative_Humidity_2m', 'Dew_Point_2m',
             'Precipitation', 'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m', 'Surface_Pressure',
             'is_day',
             'month', 'hour', 'WIND_capa', 'SR_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'],
    "SR": ['Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
           'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance', 'Cloud_Cover', 'Cloud_Cover_Low',
           'Cloud_Cover_Mid', 'Cloud_Cover_High', 'Temperature_2m', 'Relative_Humidity_2m', 'Dew_Point_2m',
           'Precipitation', 'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m', 'Surface_Pressure',
           'is_day',
           'month', 'hour', 'WIND_capa', 'SR_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'],
}
