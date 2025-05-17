import pandas as pd
from API.SUPABASE.client import getDfSupabase, insertDfSupabase, getAccessSupabase
from Asset_Modeling.Energy_Modeling.data.data import fetchRESGenerationData
from Logger.Logger import mylogger

def CaptureRate_tec(techno='WIND', year=2022):
    df = buildDfPriceVolume()
    VWA_price_res = VWA_price_tec(df, techno)
    baseload_prices = getDfSupabase('CALPowerPriceFR')
    baseload_prices['id'] = pd.to_datetime(baseload_prices['id'], utc=True)
    baseload_prices.index = baseload_prices['id']
    if isinstance(baseload_prices.index, pd.DatetimeIndex):
        baseload_prices = baseload_prices[baseload_prices.index.year == year-1]
    elif 'year' in baseload_prices.columns:
        baseload_prices = baseload_prices[baseload_prices['year'] == year-1]
    else:
        raise ValueError("Le DataFrame 'baseload_prices' doit avoir un index datetime ou une colonne 'year'.")

    # 5. Moyenne des prix baseload sur l’année
    avgprices = baseload_prices[str(year)].mean()

    # 6. Calcul du capture rate
    return VWA_price_res / avgprices
def CaptureRate(df:pd.DataFrame()):
    VWA_price_res = VWA_price(df)
    baseload_prices = getDfSupabase('CALPowerPriceFR')
    baseload_prices['id'] = pd.to_datetime(baseload_prices['id'], utc=True)
    baseload_prices.index = baseload_prices['id']
    yearly_capture = {}
    for year in sorted(baseload_prices.index.unique()):
        if isinstance(baseload_prices.index, pd.DatetimeIndex):
            baseload_prices = baseload_prices[baseload_prices.index.year == year - 1]
        elif 'year' in baseload_prices.columns:
            baseload_prices = baseload_prices[baseload_prices['year'] == year - 1]
        else:
            raise ValueError("Le DataFrame 'baseload_prices' doit avoir un index datetime ou une colonne 'year'.")

        # 5. Moyenne des prix baseload sur l’année
        avgprices = baseload_prices[str(year)].mean()
        yearly_capture[str(year)] = {'average price': avgprices, 'VWA price': VWA_price_res, 'Capture rate':VWA_price_res / avgprices}
    return yearly_capture
def VWA_price_tec(df, techno='WIND'):
    VWA_price = (df[techno] * df['price']).sum() / df[techno].sum() if df[techno].sum() != 0 else None
    return VWA_price

def VWA_price(df):
    VWA_price = (df['generation'] * df['price']).sum() / df['generation'].sum() if df['generation'].sum() != 0 else None
    return VWA_price
def AVG_price(df):
    AVG_price = df['price'].mean()
    return AVG_price

def buildDfPriceVolume():
    df_price = getDfSupabase('DAPowerPriceFR')
    df_generation = fetchRESGenerationData()
    df_price['id'] = pd.to_datetime(df_price['id'], utc=True)
    df_price.index = df_price['id']
    df = pd.concat([df_price, df_generation], axis=1)
    df = df[['SR', 'WIND', 'price']]
    df = df.dropna()
    return df

def computeShapeRate(df, techno='WIND'):
    return (VWA_price_tec(df, techno) / AVG_price(df) * 100)

def shapeYearly(df, techno='WIND'):
    shape_dict = {}
    for year in range(2017, 2026):
        df_filtered = df[df.index.year == year]
        shape_dict[year] = computeShapeRate(df_filtered, techno)
    return pd.DataFrame(list(shape_dict.items()), columns=['year', 'value'])


def shapeMonthly(df, techno='WIND'):
    shape_dict = {}
    for year in range(2017, 2026):
        for month in range(1, 13):
            df_filtered = df[(df.index.year == year) & (df.index.month == month)]
            if not df_filtered.empty:
                shape_dict[(year, month)] = computeShapeRate(df_filtered, techno)

    # Convert to DataFrame
    df_shape = pd.DataFrame(
        [(year, month, value) for (year, month), value in shape_dict.items()],
        columns=['year', 'month', 'value']
    )
    return df_shape

def SaveShape(techno, shapes):
    if not techno:
        mylogger.logger.warning('Techno parameter not set. Enter WIND or SR.')
        return
    else:
        mylogger.logger.info('Saving Shape starting...')
        db = 'ShapeMonthSR_FR' if techno=='SR' else 'ShapeMonthWIND_FR'
        mylogger.logger.info(f'Database found : {db}')

        shapes['datetime'] = pd.to_datetime(shapes[['year', 'month']].assign(day=1))
        shapes_wind['datetime'] = shapes_wind['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        shapes = shapes.drop(columns=['year', 'month'])
        supabase = getAccessSupabase(db)
        shapes_data = shapes.to_dict(orient="records")
        mylogger.logger.info('Database updating...')
        supabase.table(db).upsert(shapes_data).execute()
        mylogger.logger.info('Database updated.')

def computeCaptureRate(techno:str, year:int):
    df = buildDfPriceVolume()
    baseload_prices = getDfSupabase('CALPowerPriceFR')

if __name__ == '__main__':
    # capturerates = {
    #     tec: {str(y): CaptureRate(tec, y) * 100 for y in [2022, 2023, 2024, 2025]}
    #     for tec in ['WIND', 'SR']
    # }
    # CaptureRate(techno='WIND', year=2022)

    df = buildDfPriceVolume()
    shapes_wind = shapeMonthly(df, 'WIND')
    # SaveShape('WIND', shapes_wind)
    shapes_solar = shapeMonthly(df, 'SR')
    shapes_solar
    shapes_solar_pivot = shapes_solar.pivot(index='year', columns='month', values='value')
    shapes_solar_pivot = shapes_solar_pivot.reindex(columns=range(1, 13))
    shapes_wind_pivot = shapes_wind.pivot(index='year', columns='month', values='value')
    shapes_wind_pivot = shapes_wind_pivot.reindex(columns=range(1, 13))

    shapes_wind_pivot['year'] = shapes_wind_pivot.index
    shapes_solar_pivot['year'] = shapes_solar_pivot.index

    shapes_wind_pivot.columns = ['January', 'February', 'March', 'April', 'May', 'June',
                                 'July', 'August', 'September', 'October', 'November', 'December', 'Year'
                                 ]
    shapes_solar_pivot.columns = ['January', 'February', 'March', 'April', 'May', 'June',
                                  'July', 'August', 'September', 'October', 'November', 'December', 'Year'
                                  ]

    print('end.')
