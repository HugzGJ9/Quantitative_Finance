# @dataclass(frozen=True)

import pandas as pd
from API.OPENMETEO.Config_class import Config, url
from Logger.Logger import mylogger
import requests
from requests.adapters import HTTPAdapter, Retry

def _requests_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.3)))
    return s

def getWeatherData(cfg: Config, data_type:str) -> pd.DataFrame:
    with _requests_session() as s:
        r = s.get(url[data_type], timeout=30); r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"shortwave_radiation": "solar_radiation"})
    df.columns = ['time', 'Solar_Radiation', 'Direct_Radiation',
       'Diffuse_Radiation', 'Direct_Normal_Irradiance',
       'Global_Tilted_Irradiance', 'Cloud_Cover', 'Cloud_Cover_Low',
       'Cloud_Cover_Mid', 'Cloud_Cover_High', 'Temperature_2m',
       'Relative_Humidity_2m', 'Dew_Point_2m', 'Precipitation',
       'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m',
       'Surface_Pressure']
    return df.assign(time=lambda d: pd.to_datetime(d.time, utc=True)).set_index("time")