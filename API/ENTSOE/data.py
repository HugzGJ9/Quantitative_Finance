from entsoe import EntsoePandasClient
import pandas as pd
from API.ENTSOE.client import CLIENT

def getPriceDaHist(country='FR', start=pd.Timestamp("2017-04-01", tz="Europe/Paris"), end=pd.Timestamp("2025-04-01", tz="Europe/Paris")):
    df = pd.DataFrame(CLIENT.query_day_ahead_prices(
        country_code=country,
        start=start,
        end=end
    ))
    df.columns = ['price']
    return df
def getGenerationData(country='FR', start=pd.Timestamp("2017-04-01", tz="Europe/Paris"), end=pd.Timestamp("2025-04-01", tz="Europe/Paris")):
    df = CLIENT.query_generation(
        country_code=country,
        start=start,
        end=end
    )
    df = df.fillna(0.0)
    df = df.resample('H').mean()
    df = df.loc[:, df.columns.get_level_values(1) == "Actual Aggregated"]
    df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Wind Offshore": "WOF", "Wind Onshore": "WON", "Solar": "SR"})
    try:
        df['WIND'] = df['WON'] + df['WOF']
    except KeyError:
        df['WIND'] = df['WON']
    return df

def getInstalledCapacityData(country='FR', start=pd.Timestamp("2017-04-01", tz="Europe/Paris"), end=pd.Timestamp("2025-04-01", tz="Europe/Paris")):
    df = CLIENT.query_installed_generation_capacity(
        country_code=country,
        start=start,
        end=end,
        psr_type=None
    )
    df = df.ffill()
    df = df.ffill().fillna(0.0)
    df.columns = ["Biomass", "Gas", "Coal", "Oil", "Geothermal", "Hydro_Pumped_Storage", "Hydro_Run_of_river", "Hydro_Water_Reservoir", "Marine", "Nuclear", "Other", "Solar", "Waste", "WOF", "WON"]
    return df
def _get(psr: str) -> pd.Series:
    start = pd.Timestamp('20150101T0001', tz='Europe/Paris')
    end = pd.Timestamp('20250417T2359', tz='Europe/Paris')
    s = CLIENT.query_generation(
        country_code="FR", start=start, end=end, psr_type=psr
    )
    s = s.tz_convert("UTC")  # guarantee UTC
    return s.resample("1H").mean().rename(psr)
if __name__ == '__main__':
    start = pd.Timestamp("2025-04-01", tz="Europe/Paris")
    end = pd.Timestamp("2025-04-10", tz="Europe/Paris")
    # getGenerationData(start=start, end=end)
    getPriceDaHist(start=start, end=end)