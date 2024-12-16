import numpy as np
import pandas as pd
import datetime

from yfinance.exceptions import YFNotImplementedError

from Options import Book_class
from Options import Options_class
from Asset_Modeling import Asset_class
from Volatility.Volatility import volatilityReport
import yfinance as yf
from Logger import Logger
booking_logg = Logger.LOGGER()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
SMILE = pd.read_excel('Volatility/Smile.xlsx', sheet_name='smile_NG')

class Booking_Request():
    def __init__(self, Book:Book_class.Book=None, Option:Options_class.Option_eu=None,
                 Asset:Asset_class.asset_BS=None):
        self.tobebooked = Book if Book is not None else (
            Option if Option is not None else (Asset if Asset is not None else None))
        self.loggerinit()
    def run_Booking(self, lot_size:int=None, book_name:str=None):
        booking_logg.logger.debug(f'Booking initalisation.')
        if lot_size is None:
            booking_logg.logger.warning('Lot size not set!')
            booking_logg.logger.debug('Please enter lot size value :')
            lot_size = input()
            booking_logg.logger.debug('Lot size now set.')
            booking_logg.logger.info(f'{lot_size=}')
        else:
            booking_logg.logger.info(f'{lot_size=}')

        if book_name is None:
            booking_logg.logger.warning('Book name not set!')
            booking_logg.logger.debug('Please enter book name :')
            book_name = input()
            booking_logg.logger.debug('Book name now set.')
            booking_logg.logger.info(f'{book_name=}')

        else:
            booking_logg.logger.info(f'{book_name=}')

        booked = self.tobebooked.run_Booking(lot_size, book_name)
        booking_logg.logger.info(f'{booked}')
        booking_logg.logger.debug(f'Booking ended.')
        return
    def loggerinit(self):
        booking_logg.logger.debug(f'Booking request initalisation.')
        attribute_dict = self.tobebooked.__dict__
        booking_logg.logger.info(f"Booking request type : {attribute_dict['type']}")
        booking_logg.logger.info(f"Booking request position : {attribute_dict['position']}")
        booking_logg.logger.info(f"Booking request Underlying price : {attribute_dict['asset'].St}")
        booking_logg.logger.info(f"Booking request strike : {attribute_dict['K']}")
        booking_logg.logger.info(f"Booking request maturity : {attribute_dict['T'] * 365}")
        booking_logg.logger.info(f"Booking request rate : {attribute_dict['r']}")
        booking_logg.logger.info(f"Booking request implied volatily : {attribute_dict['sigma']}")
        booking_logg.logger.info(f"Booking request barrier : {attribute_dict['barrier']}")
        return

def run_Mtm(VI, LS, book_name=None):
    booking_logg.logger.debug(f'Start Mark to Market.')

    if book_name is None:
        booking_logg.logger.warning('book name not set!')
        booking_logg.logger.debug('Please enter book name :')
        book_name = input()
        booking_logg.logger.debug('Book name now set.')
        booking_logg.logger.info(f'{book_name=}')
    else:
        booking_logg.logger.info(f'{book_name=}')

    booking_logg.logger.info(f"Implied volatily = {VI}.xlsx")
    booking_logg.logger.info(f"Lot size = {LS}.xlsx")

    booking_file_path = f"Booking/{book_name}.xlsx"

    booking_file_sheet_name = 'histo_order'

    df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
    if not len(df['asset'].unique()) == 1:
        booking_logg.logger.critical('Multiple assets within the book.')
        return
    else:
        asset_ticker = df['asset'].unique()[0]
    ng2_ticker = asset_ticker
    try:
        ng2_data = yf.Ticker(ng2_ticker)
        ng2_price = ng2_data.history(period='1d')['Close'].iloc[0]
    except IndexError as e:
        booking_logg.logger.critical(f"Error: {e}. Ticker used not found.")
        booking_logg.logger.critical(f"End Mark to Market.")
        return
    except Exception as e:
        booking_logg.logger.critical(f"An unexpected error occurred: {e}")
        booking_logg.logger.critical(f"End Mark to Market.")
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
            option = Options_class.Option_eu(position.quantité, position.type, asset, position.strike, time2matu/365, 0.1, VI)
            list_of_positions.append(option)

    book = Book_class.Book(list_of_positions)
    book.clean_basket()
    MtM = pd.DataFrame(columns=['open position', 'type', 'time to maturity', 'strike', 'quantity', 'asset', 'asset price', 'MtM', 'moneyness %', 's-p', 'pnl', 'opnl', 'delta', 'gamma', 'vega', 'theta'])
    for i in range(len(book.basket)):

        MtM.loc[i] = {'open position':'long' if book.basket[i].position>0 else 'short', 'type': book.basket[i].type, 'time to maturity':book.basket[i].T*365, 'strike':book.basket[i].K, 'quantity': book.basket[i].position, 'asset':book.basket[i].asset.name, 'asset price':book.basket[i].asset.St, 'MtM':book.basket[i].option_price_close_formulae()*LS, 'moneyness %': (book.basket[i].asset.St/book.basket[i].K - 1)*100, 's-p':None, 'pnl': None, 'opnl':None, 'delta':book.basket[i].Delta_DF(), 'gamma':book.basket[i].Gamma_DF(), 'vega':book.basket[i].Vega_DF(), 'theta': book.basket[i].Theta_DF()}

    MtM.loc[len(MtM)] = {'open position': 'long' if asset.quantity>0 else 'short',
                        'type': 'Asset',
                        'quantity': asset.quantity, 'asset': asset.name,
                        'asset price': asset.St, 'MtM': asset.St*asset.quantity*LS,
                        'delta': asset.Delta_DF(), 'gamma': None,
                        'vega': None, 'theta': None}

    Mtm_book=book.option_price_close_formulae()*LS
    sp_book = df['s-p'].sum()
    MtM.loc[len(MtM)] = {'open position': 'long' if book.asset.quantity>0 else 'short',
                        'type': 'Book',
                        'time to maturity': None,
                        'strike': None,
                        'quantity': None, 'asset': None,
                        'asset price': book.asset.St, 'MtM': Mtm_book,
                        's-p': sp_book, 'pnl': Mtm_book+sp_book,
                        'opnl': None,
                        'delta': book.Delta_DF(), 'gamma': book.Gamma_DF(),
                        'vega': book.Vega_DF(), 'theta': book.Theta_DF()}
    with pd.ExcelWriter(booking_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        MtM.to_excel(writer, sheet_name='MtM', index=False)
        booking_logg.logger.debug(f"Booking file updated.")

    booking_logg.logger.info("\n%s", MtM.to_string())
    booking_logg.logger.debug(f"End Mark to Market.")
    return


if __name__ == '__main__':

    stock = Asset_class.asset_BS(3.268, 0, "NG=F")
    # option = Options_class.Option_eu(-10, 'Call EU', stock, 2.5, 2/365, 0.1, 0.7)
    option = Options_class.Option_eu(1, 'Call EU', stock, 3.268, 7/365, 0.1, volatility_surface_df=SMILE, use_vol_surface=True)
    option.GammaRisk()
    option.Skew_DF()
    book_name = "GasCall"
    volatilityReport()
    # book = Book_class.Book([option])
    #

    # print(book.option_price_close_formulae())
    # book.RiskAnalysis()
    # print(book.Delta_DF())
    # df_pnl = book.PnlRisk()
    # # option.Vega_surface()
    # # option.Theta_surface()
    # #
    # book.get_move_deltahedge()
    # # # #
    # #
    # booking_request = Booking_Request(option)
    # booking_request.run_Booking(lot_size=10000, book_name=book_name)
    # run_Mtm(VI=0.62, LS=10000, book_name=book_name)