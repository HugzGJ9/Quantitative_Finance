import matplotlib.pyplot as plt
from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu, Option_prem_gen, plot_greek_curves_2d
from Options.Book_class import Book
from Logger.Logger import mylogger

# from interest_rates import Tresury_bond_13weeks
# from interest_rates import Tresury_bond_5years
# from interest_rates import Tresury_bond_30years
if __name__ == '__main__':

    stock1 = asset_BS(100, 0)
    option1 = Option_eu(-1, 'Put EU', stock1, 99, 10 / 365, 0.02, 0.50)
    option2 = Option_eu(1, 'Put EU', stock1, 101, 10 / 365, 0.02, 0.50)
    book1 = Book([option1, option2])
    mylogger.logger.info(f"Book value = {book1.option_price_close_formulae()}")
    book1.RiskAnalysis()