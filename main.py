from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from Asset_class import asset_BS
from Options_class import Option_eu, Option_prem_gen, plot_greek_curves_2d
from Book_class import Book
# from interest_rates import Tresury_bond_13weeks
# from interest_rates import Tresury_bond_5years
# from interest_rates import Tresury_bond_30years

if __name__ == '__main__':
    Nmc = 100
    N = 5
    t = 0
    K = 100
    vol = 0.2
    S0 = 100
    r = 0.1
    maturity_date = '2024-12-27'
    now = datetime.now()
    T = pd.to_datetime(maturity_date) - pd.to_datetime(now)
    T = T.total_seconds()/(3600*365*24)   # maturity_dates = ['2023-12-15', '2023-12-22', '2023-12-29', '2024-05-01', '2024-12-01', '2025-12-01']
    stock_name = 'BTC-USD'
    # r = Tresury_bond_13weeks
    # for maturity_date in maturity_dates:
    #implied_vol_dict = Volatilite_implicite(stock, maturity_date, 'Call EU', r, True, True)
    # plt.show()
    # for maturity_date in maturity_dates:
    #     implied_vol_dict = Volatilite_implicite(stock, maturity_date, 'Put EU', r, False)
    # plt.show()
    # print(Tresury_bond_13weeks)
    #
    # callEU = Option_eu(1, 'Call EU', 100, 95, 0, T, r, vol)
    # PutEU = Option_eu(1, 'Put EU', 100, 95, 0, T, r, vol)
    #
    # plot_greek_curves_2d(1, 'Strangle', 'Delta', [50, 120], t, T, r, vol)
    # plt.show()
    #
    # position1.append(PutEU)
    # T = 1/(365.6*2)
    vol_implied = 0.58
    strike = 43000
    stock_obj = yf.Ticker(stock_name)
    S0 = stock_obj.history().tail(1)['Close'].values[0]
    S0 = 46576
    stock1 = asset_BS(S0, 0)
    callEU = Option_eu(1, 'Call EU', stock1, strike, 0, T, r, vol_implied)
    #callEU2 = Option_eu(-2, 'Call EU', stock1, 135, 0, T, r, vol)
    book1 = Book([callEU])

    #book1.simu_asset(1)
    strangle = Option_prem_gen(-1, 'Strangle', stock1, [95, 105], 0, T, r, vol)
    strangle.display_payoff_option()
    print('book greeks')
    print(book1.Delta_DF())
    print(book1.Gamma_DF())
    print(book1.Theta_DF())
    print(f'book price = {book1.option_price_close_formulae()}')
    print(f'stock price = {book1.basket[0].St}')
    book1.simu_asset(1)
    print('asset evolution done')
    print(f'book price = {book1.option_price_close_formulae()}')
    print(f'stock price = {book1.basket[0].St}')
    print(book1.Delta_DF())
    print(book1.Gamma_DF())
    print(book1.Theta_DF())

    plot_greek_curves_2d(-1, 'Call EU', 'Theta', 95, t, T, r, vol)
    plt.show()

    straddle = Option_prem_gen(1, 'Strangle', stock1, [95, 95], 0, T, r, vol)

    straddle.option_price_close_formulae()
    stock1.simu_asset(1)

    T = [0.1, 0.2, 1]

    plot_greek_curves_2d(-1, 'Strangle', 'Delta', [50, 120], t, T, r, vol)
    plt.show()
    plot_greek_curves_2d(-1,'Strangle', 'Gamma', [50, 120], t, T, r, vol)
    plt.show()
    plot_greek_curves_2d(-1,'Strangle', 'Vega', [50, 120], t, T, r, vol)
    plt.show()
    plot_greek_curves_2d(-1,'Strangle', 'Theta', [50, 120], t, T, r, vol)
    plt.show()
    #to activate the user interface
    # root = ThemedTk(theme="breeze")
    # root.mainloop()

      # Choose your preferred theme

    #to activate the user interface
    # root = ThemedTk(theme="breeze")
    # root.mainloop()

      # Choose your preferred theme