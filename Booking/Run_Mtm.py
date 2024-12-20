import datetime
import warnings

import pandas as pd
import yfinance as yf

from Asset_Modeling import Asset_class
from Logger.Logger import mylogger
from Options import Options_class, Book_class
SMILE = pd.read_excel('../Volatility/Smile.xlsx', sheet_name='smile_NG')



def run_Mtm(LS, book_name=None):
    mylogger.logger.debug(f'Start Mark to Market.')

    if book_name is None:
        mylogger.logger.warning('book name not set!')
        mylogger.logger.debug('Please enter book name :')
        book_name = input()
        mylogger.logger.debug('Book name now set.')
        mylogger.logger.info(f'{book_name=}')
    else:
        mylogger.logger.info(f'{book_name=}')

    mylogger.logger.info(f"Lot size = {LS}.xlsx")

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
    MtM = pd.DataFrame(columns=['open position', 'type', 'time to maturity', 'strike', 'quantity', 'asset', 'asset price', 'MtM', 'moneyness %', 's-p', 'pnl', 'opnl', 'delta', 'gamma', 'vega', 'theta'])
    for i in range(len(book.basket)):

        MtM.loc[i] = {'open position':'long' if book.basket[i].position>0 else 'short', 'type': book.basket[i].type, 'time to maturity':book.basket[i].T*365, 'strike':book.basket[i].K, 'quantity': book.basket[i].position, 'asset':book.basket[i].asset.name, 'asset price':book.basket[i].asset.St, 'MtM':book.basket[i].option_price_close_formulae()*LS, 'moneyness %': (book.basket[i].asset.St/book.basket[i].K - 1)*100 if book.basket[i].type=='Call EU' else -(book.basket[i].asset.St/book.basket[i].K - 1)*100, 's-p':None, 'pnl': None, 'opnl':None, 'delta':book.basket[i].Delta_DF(), 'gamma':book.basket[i].Gamma_DF(), 'vega':book.basket[i].Vega_DF(), 'theta': book.basket[i].Theta_DF()}

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
        mylogger.logger.debug(f"Booking file updated.")

    mylogger.logger.info("\n%s", MtM.to_string())
    mylogger.logger.debug(f"End Mark to Market.")
    return

if __name__ == '__main__':
    run_Mtm(LS=10000, book_name="GasCall")
    run_Mtm(LS=10000, book_name="Skew")