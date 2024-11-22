import numpy as np
import pandas as pd

from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu, Option_prem_gen, plot_greek_curves_2d
from Options.Book_class import Book
from Logger.Logger import mylogger
SMILE = pd.read_excel('Volatility/Smile.xlsx')
def runDemo():
    stock1 = asset_BS(100, 0, logger=True)
    option1 = Option_eu(-1, 'Put EU', stock1, 99, 10 / 365, 0.02, sigma=0.37, use_vol_surface=False, logger=True)
    option2 = Option_eu(1, 'Put EU', stock1, 101, 10 / 365, 0.02, sigma=0.37, use_vol_surface=False, logger=True)
    book1 = Book([option1, option2], logger=True)
    mylogger.logger.info(f"Book value = {book1.option_price_close_formulae()}")
    book1.RiskAnalysis(logger=True)
    book1.display_payoff_option()
    book1.PnlRisk()
    book1.VannaRisk()
    book1.VolgaRisk()


if __name__ == '__main__':
    runDemo()
