import numpy as np
import pandas as pd

from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu, Option_prem_gen
from Options.Book_class import Book
from Logger.Logger import mylogger
SMILE = pd.read_excel('Volatility/Smile.xlsx', sheet_name='smile')
def runDemo():
    stock1 = asset_BS(100, 0, logger=True)
    option1 = Option_eu(position=100, type='Call EU', asset=stock1, K=100, T=3 / 365, r=0.02, sigma=0.5, use_vol_surface=False, logger=True)
    option2 = Option_eu(position=-100, type='Call EU', asset=stock1, K=105, T=3 / 365, r=0.02, sigma=0.5, use_vol_surface=False, logger=True)

    book1 = Book([option1, option2], logger=True)

    book_price = book1.option_price_close_formulae()
    mylogger.logger.info(f"Book Value : {book_price}")
    book1.RiskAnalysis(logger=True)
    book1.PnlRisk()
    book1.display_payoff_option()
    mylogger.logger.info(f"Book delta = {book1.Delta_DF()}")
    book1.delta_hedge(logger=True)
    mylogger.logger.info(f"Book delta = {book1.Delta_DF()}")
    book1.PnlRisk()
    book1.display_payoff_option()
    return


def runDemoVolSurface():

    stock1 = asset_BS(100, 0)
    option1 = Option_eu(position=100, type='Call EU', asset=stock1, K=100, T=3 / 365, r=0.02, volatility_surface_df=SMILE, use_vol_surface=True)
    option2 = Option_eu(position=-100, type='Call EU', asset=stock1, K=105, T=3 / 365, r=0.02, volatility_surface_df=SMILE, use_vol_surface=True)

    book1 = Book([option1, option2], logger=True)

    book_price = book1.option_price_close_formulae()
    mylogger.logger.info(f"Book Value : {book_price}")
    book1.RiskAnalysis(logger=True)
    book1.PnlRisk()
    book1.display_payoff_option()
    mylogger.logger.info(f"Book delta = {book1.Delta_DF()}")
    book1.delta_hedge(logger=True)
    mylogger.logger.info(f"Book delta = {book1.Delta_DF()}")
    book1.PnlRisk()
    book1.display_payoff_option()
    return


if __name__ == '__main__':
    # runDemo() #consider the volatility as a constant.
    runDemoVolSurface() #consider the volatility as a surface depending on time and moneyness. Volatility surface is describ within the file Volatility/Smile.xlsx
