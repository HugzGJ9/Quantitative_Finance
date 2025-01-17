import matplotlib.pyplot as plt
import pandas as pd

from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu, Option_prem_gen
from Options.Book_class import Book
from Logger.Logger import mylogger

SMILE = pd.read_excel('Quantitative_Finance/Volatility/Smile.xlsx', sheet_name='smile_NG')

# from interest_rates import Tresury_bond_13weeks
# from interest_rates import Tresury_bond_5years
# from interest_rates import Tresury_bond_30years
if __name__ == '__main__':

    stock1 = asset_BS(100, 0)
    option1 = Option_eu(1, 'Call EU', stock1, 120, 6 / 365, 0.02,  volatility_surface_df=SMILE, use_vol_surface=True)
    option1.show_volatility_surface()
    option2 = Option_eu(1, 'Call EU', stock1, 100, 6 / 365, 0.02,  volatility_surface_df=SMILE, use_vol_surface=True)
    book1 = Book([option1, option2])
    option1.find_vol_impli(0.427)
    option2.find_vol_impli(0.021)

    book1 = Book([option1])
    mylogger.logger.info(f"Book value = {book1.option_price_close_formulae()}")
    book1.RiskAnalysis()