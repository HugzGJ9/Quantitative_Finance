from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu, Option_prem_gen
from Options.Book_class import Book
from Logger.Logger import mylogger
from Volatility.Volatility_surface import Vol_surface

if __name__ == '__main__':

    stock1 = asset_BS(100, 0)
    option1 = Option_eu(1, 'Call EU', stock1, 120, 6 / 365, 0.02, volatility_surface_df=Vol_surface, use_vol_surface=True)
    option1.show_volatility_surface()
    option2 = Option_eu(1, 'Call EU', stock1, 100, 6 / 365, 0.02, volatility_surface_df=Vol_surface, use_vol_surface=True)
    book1 = Book([option1, option2])

    book1 = Book([option1])
    mylogger.logger.info(f"Book value = {book1.option_price_close_formulae()}")
    book1.RiskAnalysis()