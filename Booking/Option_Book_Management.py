import datetime

import pandas as pd
import yfinance as yf

from Asset_Modeling import Asset_class
from Logger.Logger import mylogger
from Options import Options_class, Book_class
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
SMILE = pd.read_excel('../Volatility/Smile.xlsx', sheet_name='smile_NG')

def importBook(book_name=None):
    booking_file_path = f"../Booking/{book_name}.xlsx"

    booking_file_sheet_name = 'histo_order'

    df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
    if not len(df['asset'].unique()) == 1:
        mylogger.logger.critical('Multiple assets within the book.')
        return
    else:
        asset_ticker = df['asset'].unique()[0]
    ng2_ticker = asset_ticker
    try:
        ng2_data = yf.Ticker(ng2_ticker)
        ng2_price = ng2_data.history(period='1d')['Close'].iloc[0]
    except IndexError as e:
        mylogger.logger.critical(f"Error: {e}. Ticker used not found.")
        mylogger.logger.critical(f"End Mark to Market.")
        return
    except Exception as e:
        mylogger.logger.critical(f"An unexpected error occurred: {e}")
        mylogger.logger.critical(f"End Mark to Market.")
        return
    list_of_positions = []
    asset = Asset_class.asset_BS(ng2_price, 0, ng2_ticker)
    for i in range(len(df)):
        position = df.loc[i]
        if position.type == 'asset':
            asset.quantity = asset.quantity+position.quantité
        else:
            delta = datetime.datetime.now().date() - pd.to_datetime(position['date heure']).date()
            time2matu=position.maturité - delta.total_seconds() / (24 * 3600)
            option_temp = Options_class.Option_eu(position.quantité, position.type, asset, position.strike, time2matu/365, 0.1, volatility_surface_df=SMILE, use_vol_surface=True)
            option = Options_class.Option_eu(position.quantité, position.type, asset, position.strike, time2matu/365, 0.1, option_temp.get_sigma())
            list_of_positions.append(option)

    book = Book_class.Book(list_of_positions)
    book.clean_basket()

    return book


if __name__ == '__main__':
    book_skew = importBook('Skew')
    book_GasCall = importBook('GasCall')
    print('end')