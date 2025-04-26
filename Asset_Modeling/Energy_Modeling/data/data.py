import pandas as pd

from API.SUPABASE.client import getDfSupabase
def fetchRESGenerationData(country="FR"):
    if country == "FR":
        df = getDfSupabase('GenerationFR')
    else:
        df = getDfSupabase('GenerationFR')
    df['id'] = pd.to_datetime(df['id'], utc=True)
    df.index = df['id']
    df.index.name = 'time'
    df = df[['SR', 'WIND']]
    res_capacity = fetchRESCapacityData(country)
    res_capacity = res_capacity.rename(columns={'SR': 'SR_capa', 'WIND': 'WIND_capa'})
    df['SR_capa'] = df.index.year.map(res_capacity['SR_capa'])
    df['WIND_capa'] = df.index.year.map(res_capacity['WIND_capa'])
    df = df[(df['SR'] <= df['SR_capa']) & (df['WIND'] <= df['WIND_capa'])]
    df = df.drop(columns=['SR_capa', 'WIND_capa'])
    # df = df[df.index.year < 2025]
    return df

def fetchRESCapacityData(country="FR"):
    if country == "FR":
        df = getDfSupabase('InstalledCapacityFR')
    else:
        df = getDfSupabase('InstalledCapacityFR')
    df['id'] = pd.to_datetime(df['id'], utc=True)
    df.index = df['id']
    df.index = df.index.year
    df.index.name = 'time'
    df['WIND'] = df['WOF'] + df['WON']
    df = df[['Solar', 'WIND']]
    df = df.rename(columns={'Solar': 'SR'})

    return df

def fetchWeatherData(country="FR"):
    if country == "FR":
        df = getDfSupabase('WeatherFR')
    else:
        df = getDfSupabase('WeatherFR')
    df['id'] = pd.to_datetime(df['id'], utc=True)
    df = df.set_index('id')
    df.index.name = 'time'
    return df

def fetchRESGenerationMonthlyData(country="FR"):
    res_generation = fetchRESGenerationData(country)
    res_capacity = fetchRESCapacityData(country=country)
    res_capacity = res_capacity.rename(columns={'SR': 'SR_capa', 'WIND': 'WIND_capa'})
    res_generation['SR_capa'] = res_generation.index.year.map(res_capacity['SR_capa'])
    res_generation['WIND_capa'] = res_generation.index.year.map(res_capacity['WIND_capa'])

    max_SR_capa = res_generation['SR_capa'].max()
    max_WIND_capa = res_generation['WIND_capa'].max()

    res_generation['SR_normalized'] = res_generation['SR'] * (max_SR_capa / res_generation['SR_capa'])
    res_generation['WIND_normalized'] = res_generation['WIND'] * (max_WIND_capa / res_generation['WIND_capa'])

    res_generation_day = res_generation.resample('D').sum()
    res_generation_month = res_generation_day.resample('M').mean()
    if res_generation_month.index.tz is not None:
        res_generation_month.index = res_generation_month.index.tz_convert(None)
        res_generation_day.index = res_generation_day.index.tz_convert(None)
    res_generation_month.index = res_generation_month.index.to_period('M').to_timestamp()
    res_generation_day.index = res_generation_day.index.to_period('D').to_timestamp()

    res_generation_month = res_generation_month.drop(columns=['SR_capa', 'WIND_capa'])
    res_generation_day = res_generation_day.drop(columns=['SR_capa', 'WIND_capa'])

    return res_generation_month, res_generation_day
if __name__ == '__main__':
    df = fetchRESGenerationMonthlyData("FR")