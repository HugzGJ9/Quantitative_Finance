import matplotlib.pyplot as plt
from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu, Option_prem_gen
from Options.Book_class import Book
from Logger.Logger import mylogger
from Volatility.Volatility import find_vol_impli

# from interest_rates import Tresury_bond_13weeks
# from interest_rates import Tresury_bond_5years
# from interest_rates import Tresury_bond_30years
if __name__ == '__main__':

    stock1 = asset_BS(3.467, 0)
    option1 = Option_eu(1, 'Call EU', stock1, 3.35, 8 / 365, 0.02, 0.50)
    find_vol_impli(option1, 3.0)
    option2 = Option_eu(1, 'Put EU', stock1, 101, 10 / 365, 0.02, 0.50)
    book1 = Book([option1])
    mylogger.logger.info(f"Book value = {book1.option_price_close_formulae()}")
    book1.RiskAnalysis()