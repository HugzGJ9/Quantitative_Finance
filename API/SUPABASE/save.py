import pandas as pd

from API.ENTSOE.data import getGenerationData
from API.GMAIL.auto_email_template import setAutoemail
from API.RTE.data import getAuctionDaData, getAPIdata
from API.SUPABASE.client import updateDfSupabase
from Logger.Logger import mylogger
def saveDailyGeneration():
    # df = getAPIdata(APIname="Actual Generation 15min", logger=True)
    now = pd.Timestamp.now(tz='Europe/Paris')
    yesterday = now.normalize() - pd.Timedelta(days=1)
    df = getGenerationData(country='FR', start=yesterday, end=now)
    saveDailyGenerationTS(df, 'REALIZED')
    return

def saveDailyAuction():
    df = getAuctionDaData()
    df.index = df.index.tz_convert('UTC')
    try:
        updateDfSupabase(df, 'DAPowerPriceFR')
    except:
        setAutoemail(
            ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com'],
            'INFO SAVE DA AUCTION FAILED.',
            '''DA Auction failed.'''
        )
    return

def saveDailyGenerationTS(df, timeseries_label):
    df = df[['SR', 'WIND']]
    label_dict = {'HUGO' : {'SR': 'HUGO_SR', 'WIND':'HUGO_WIND'},
            'RTE': {'SR': 'RTE_SR', 'WIND':'RTE_WIND'},
            'REALIZED': {'SR': 'REALIZED_SR', 'WIND':'REALIZED_WIND'}}
    if timeseries_label not in label_dict.keys():
        mylogger.logger.error('timeseries label not found, has to be HUGO, RTE or REALIZED')
        return
    df = df.rename(columns={"SR": label_dict[timeseries_label]['SR'], "WIND": label_dict[timeseries_label]['WIND']})
    df.index = df.index.tz_convert('UTC')
    try:
        updateDfSupabase(df, 'ForecastGenerationFR')
    except:
        setAutoemail(
            ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com'],
            f'SAVE DAILY GENERATION FORECAST {timeseries_label} FAILED',
            '''SAVE FAILED.'''
        )
    return
    



