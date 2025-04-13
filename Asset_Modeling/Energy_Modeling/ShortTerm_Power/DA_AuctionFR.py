from API.RTE.data import getAPIdata
import os
import pandas as pd

def updatefileDA_Auction():
    os.chdir('../../../API')
    data = pd.read_excel("../Data/DA_PriceFR.xlsx")
    df_da_prices = getAPIdata(APIname="Wholesale Market")
    df_da_prices['date'] = pd.to_datetime(df_da_prices['date'], errors='coerce')

    df_da_prices['date'] = pd.to_datetime(
        df_da_prices['date'].astype(str) + ' ' + df_da_prices['time'].astype(str),
        format='%Y-%m-%d %H:%M:%S'
    )
    df_da_prices.drop(columns='time', inplace=True)

    data = pd.concat([data, df_da_prices], ignore_index=True)
    data = data.drop_duplicates(subset=['date'], keep='last')
    data.reset_index(drop=True, inplace=True)

    with pd.ExcelWriter("../Data/DA_PriceFR.xlsx", engine='openpyxl') as writer:
        data.to_excel(writer, index=False)
    return


if __name__ == '__main__':
    updatefileDA_Auction()