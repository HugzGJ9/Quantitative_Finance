from API.RTE.data import getAuctionDaData
from API.SUPABASE.client import insertDfSupabase

def saveAuctionData():
    df = getAuctionDaData()
    df.index = df.index.tz_convert('UTC')
    insertDfSupabase(df, 'DAPowerPriceFR')
if __name__ == '__main__':
    saveAuctionData()
