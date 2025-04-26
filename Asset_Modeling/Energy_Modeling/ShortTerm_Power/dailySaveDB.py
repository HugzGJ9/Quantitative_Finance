import pandas as pd

from API.RTE.data import getAuctionDaData
from API.ENTSOE.data import getGenerationData
from API.SUPABASE.client import updateDfSupabase
from API.GMAIL.auto_email_template import setAutoemail


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

def saveDailyGeneration():
    now = pd.Timestamp.now(tz='Europe/Paris')
    yesterday = now.normalize() - pd.Timedelta(days=1)
    df =  getGenerationData(country='FR', start=yesterday, end=now)
    df.index = df.index.tz_convert('UTC')
    try:
        updateDfSupabase(df, 'DAPowerPriceFR')
    except:
        setAutoemail(
            ['hugo.lambert.perso@gmail.com', 'hugo.lambert.perso@gmail.com'],
            'INFO SAVE DAILY GENERATION FAILED.',
            '''Daily Generation failed.'''
        )
    return

if __name__ == '__main__':
    saveDailyAuction()
    saveDailyGeneration()
