import numpy as np
from supabase import create_client
import pandas as pd

from API.SUPABASE.prices import CALPRICES
from Keypass.key_pass import API_SUPABASE
from API.ENTSOE.data import getGenerationData
from Logger.Logger import mylogger

def getAccessSupabase(App: str):
    if App in API_SUPABASE.keys():
        SUPABASE_URL = API_SUPABASE[App]['token_url']
        SUPABASE_SERVICE_KEY = API_SUPABASE[App]['key']
    else:
        mylogger.logger.error(f"APP : {App} not found in SUPABASE Database.")
        return
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return  supabase

def getDfSupabase(db_name):
    supabase = getAccessSupabase(db_name)
    df = pd.DataFrame(supabase.table(db_name).select("*").execute().data)
    return df
def getRowsSupabase(db_name, ids):
    supabase = getAccessSupabase(db_name)
    response = supabase.table(db_name).select("*").in_('id', ids).execute()
    df = pd.DataFrame(response.data)
    return df
def removeAllRowsSupabase(db_name):
    supabase = getAccessSupabase(db_name)
    data = supabase.table(db_name).select("id").execute().data
    ids_to_delete = [item['id'] for item in data]
    for row_id in ids_to_delete:
        supabase.table(db_name).delete().eq('id', row_id).execute()
    print(f"Removed {len(ids_to_delete)} rows from '{db_name}'")
    return

def insertDfSupabase(df, db_name):
    supabase = getAccessSupabase(db_name)
    df.index.name = 'id'
    data = df.reset_index()

    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    data = data.replace({np.nan: None, np.inf: None, -np.inf: None})
    data = data.where(pd.notnull(data), None)
    data = data.to_dict(orient="records")
    supabase.table(db_name).insert(data).execute()
    return
def updateDfSupabase(df, db_name):
    supabase = getAccessSupabase(db_name)
    data_db = supabase.table(db_name).select('*').in_('id', df.index).execute()
    data_db = pd.DataFrame(data_db.data)
    if not data_db.empty:

        data_db = data_db.set_index('id')
        data_db.update(df)
        data_db.index.name = 'id'
        data = data_db.reset_index()
    else:
        df.index.name = 'id'
        data = df.reset_index()
    supabase.table(db_name).delete().in_('id', df.index).execute()
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    data = data.replace({np.nan: None, np.inf: None, -np.inf: None})
    data = data.where(pd.notnull(data), None)
    data = data.to_dict(orient="records")
    supabase.table(db_name).upsert(data, ignore_duplicates=True).execute()
    return

if __name__ == '__main__':
    # for i in range(10*6):
    #     start = pd.Timestamp('20200301', tz='Europe/Paris') - pd.DateOffset(months=i*2)
    #     end = pd.Timestamp('20200601', tz='Europe/Paris') - pd.DateOffset(months=i*2)
    #     print(f"{start} - {end}")
    #     data = getGenerationData(start=start, end=end)
    #     data.index = data.index.tz_convert('UTC')
    #     updateDfSupabase(data, 'GenerationFR')

    # start = pd.Timestamp('20250401', tz='Europe/Paris')
    # end = pd.Timestamp('20250501', tz='Europe/Paris')
    # print(f"{start} - {end}")
    # data = getGenerationData(start=start, end=end)
    # data.index = data.index.tz_convert('UTC')
    # updateDfSupabase(data, 'GenerationFR')
    # removeAllRowsSupabase('GenerationFR')
    # now = pd.Timestamp.now(tz='Europe/Paris')
    # yesterday = now.normalize() - pd.Timedelta(days=1)
    # df = getGenerationData(country='FR', start=yesterday, end=now)
    df = CALPRICES
    insertDfSupabase(df, 'CALPowerPriceFR')
