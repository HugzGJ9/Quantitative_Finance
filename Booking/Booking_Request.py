import pandas as pd

from Options import Book_class
from Options import Options_class
from Asset_Modeling import Asset_class
from Logger.Logger import mylogger

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
SMILE = pd.read_excel('../Volatility/Smile.xlsx', sheet_name='smile_NG')

class Booking_Request():
    def __init__(self, Book: Book_class.Book = None, Option: Options_class.Option_eu = None,
                 Asset: Asset_class.asset_BS = None):
        if Book is not None:
            self.tobebooked = Book
        elif Option is not None:
            self.tobebooked = Option
            self.loggerinit()

        elif Asset is not None:
            self.tobebooked = Asset
        else:
            self.tobebooked = None


    def run_Booking(self, lot_size:int=None, book_name:str=None):
        mylogger.logger.debug(f'Booking initalisation.')
        if lot_size is None:
            mylogger.logger.warning('Lot size not set!')
            mylogger.logger.debug('Please enter lot size value :')
            lot_size = input()
            mylogger.logger.debug('Lot size now set.')
            mylogger.logger.info(f'{lot_size=}')
        else:
            mylogger.logger.info(f'{lot_size=}')

        if book_name is None:
            mylogger.logger.warning('Book name not set!')
            mylogger.logger.debug('Please enter book name :')
            book_name = input()
            mylogger.logger.debug('Book name now set.')
            mylogger.logger.info(f'{book_name=}')

        else:
            mylogger.logger.info(f'{book_name=}')

        booked = self.tobebooked.run_Booking(lot_size, book_name)
        mylogger.logger.info(f'{booked}')
        mylogger.logger.debug(f'Booking ended.')
        return
    def loggerinit(self):
        mylogger.logger.debug(f'Booking request initalisation.')
        attribute_dict = self.tobebooked.__dict__
        mylogger.logger.info(f"Booking request type : {attribute_dict['type']}")
        mylogger.logger.info(f"Booking request position : {attribute_dict['position']}")
        mylogger.logger.info(f"Booking request Underlying price : {attribute_dict['asset'].St}")
        mylogger.logger.info(f"Booking request strike : {attribute_dict['K']}")
        mylogger.logger.info(f"Booking request maturity : {attribute_dict['T'] * 365}")
        mylogger.logger.info(f"Booking request rate : {attribute_dict['r']}")
        mylogger.logger.info(f"Booking request implied volatily : {attribute_dict['sigma']}")
        mylogger.logger.info(f"Booking request barrier : {attribute_dict['barrier']}")
        return


if __name__ == '__main__':

    stock = Asset_class.asset_BS(3.62, -1, "NG=F")
    option = Options_class.Option_eu(1, 'Call EU', stock, 3.30, 6/365, 0.1, 0.85)
    option2 = Options_class.Option_eu(-10, 'Call EU', stock, 3.6, 6/365, 0.1, 0.83)
    option3 = Options_class.Option_eu(1, 'Call EU', stock, 3.30, 6/365, 0.1, 0.75)

    book = Book_class.Book([option])

    book_name = "GasCall"
    book_name = "Skew"
    # volatilityReport()

    booking_request = Booking_Request(stock)
    booking_request.run_Booking(lot_size=10000, book_name=book_name)
