import pandas as pd
import datetime
from Options import Book_class
from Options import Options_class
from Asset_Modeling import Asset_class
import yfinance as yf
class Booking_Request():
    def __init__(self, Book:Book_class.Book=None, Option:Options_class.Option_eu=None,
                 Asset:Asset_class.asset_BS=None):
        self.tobebooked = Book if Book is not None else (
            Option if Option is not None else (Asset if Asset is not None else None))
    def run_Booking(self, lot_size:int=None):
        self.tobebooked.run_Booking(lot_size)
        return

def run_Mtm(VI, LS):
    booking_file_path = 'Booking_history.xlsx'
    booking_file_sheet_name = 'histo_order'
    ng2_ticker = "NG=F"  # The continuous contract for Natural Gas
    ng2_data = yf.Ticker(ng2_ticker)
    ng2_price = ng2_data.history(period='1d')['Close'].iloc[0]
    list_of_positions = []
    asset = Asset_class.asset_BS(ng2_price, 0, "HH NG2!")
    df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
    for i in range(len(df)):
        position = df.loc[i]
        if position.type == 'asset':
            asset.quantity = asset.quantity+position.quantité
        else:
            delta = datetime.datetime.now().date() - pd.to_datetime(position['date heure']).date()
            time2matu=position.maturité - delta.total_seconds() / (24 * 3600)
            option = Options_class.Option_eu(position.quantité, position.type, asset, position.strike, time2matu/365.6, 0.1, VI)
            list_of_positions.append(option)

    book = Book_class.Book(list_of_positions)
    book.clean_basket()
    MtM = pd.DataFrame(columns=['open position', 'type', 'time to maturity', 'strike', 'quantity', 'asset', 'asset price', 'MtM', 'moneyness %', 's-p', 'pnl', 'opnl', 'delta', 'gamma', 'vega', 'theta'])
    for i in range(len(book.basket)):

        MtM.loc[i] = {'open position':'long' if book.basket[i].position>0 else 'short', 'type': book.basket[i].type, 'time to maturity':book.basket[i].T*365.6, 'strike':book.basket[i].K, 'quantity': book.basket[i].position, 'asset':book.basket[i].asset.name, 'asset price':book.basket[i].asset.St, 'MtM':book.basket[i].option_price_close_formulae()*LS, 'moneyness %': (book.basket[i].asset.St/book.basket[i].K - 1)*100, 's-p':None, 'pnl': None, 'opnl':None, 'delta':book.basket[i].Delta_DF(), 'gamma':book.basket[i].Gamma_DF(), 'vega':book.basket[i].Vega_DF(), 'theta': book.basket[i].Theta_DF()}

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
    return


if __name__ == '__main__':

    stock = Asset_class.asset_BS(2.34, -2, "HH NG2!")
    option = Options_class.Option_eu(10, 'Call EU', stock, 2.5, 6/365.6, 0.1, 0.6)
    book = Book_class.Book([option])
    print(book.option_price_close_formulae())
    book.RiskAnalysis()
    print(book.Delta_DF())
    df_pnl = book.PnlRisk()
    # option.Vega_surface()
    # option.Theta_surface()
    #
    book.get_move_deltahedge()
    # # # #
    # #
    # booking_request = Booking_Request(stock)
    # booking_request.run_Booking(lot_size=10000)
    # run_Mtm(VI=0.6, LS=10000)