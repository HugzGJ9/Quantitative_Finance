import pandas as pd

from Options import Book_class
from Options import Options_class
from Asset_Modeling import Asset_class
import yfinance as yf
class Booking():
    def __init__(self, Book:Book_class.Book=None, Option:Options_class.Option_eu=None,
                 Asset:Asset_class.asset_BS=None):
        self.tobebooked = Book if Book is not None else (
            Option if Option is not None else (Asset if Asset is not None else None))
    def run_Booking(self):
        self.tobebooked.run_Booking()
        return

def run_Mtm():
    booking_file_path = 'Booking_history.xlsx'
    booking_file_sheet_name = 'histo_order'
    ng2_ticker = "NGX24.NYM"  # The continuous contract for Natural Gas
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
            option = Options_class.Option_eu(position.quantité, position.type, asset, position.strike, position.maturité, 0.1, position.vol)
            list_of_positions.append(option)

    book = Book_class.Book(list_of_positions)
    MtM = pd.DataFrame(columns=['open position', 'quantity', 'asset', 'asset price', 'MtM', 'delta', 'gamma', 'vega', 'theta'])
    # for i in range(len(book.basket)):
    #     MtM.loc(i)=
    return
    # with pd.ExcelWriter(booking_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    #     df.to_excel(writer, sheet_name=booking_file_sheet_name, index=False)

if __name__ == '__main__':
    stock = Asset_class.asset_BS(100, 0, "HH NG2!")
    option = Options_class.Option_eu(100, 'Call EU', stock, 105, 1, 0.1, 0.6)
    booking_request = Booking(option)

    # booking_request.run_Booking()
    run_Mtm()