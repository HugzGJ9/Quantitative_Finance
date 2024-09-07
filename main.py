import matplotlib.pyplot as plt
from  Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu, Option_prem_gen, plot_greek_curves_2d
from Options.Book_class import Book
# from interest_rates import Tresury_bond_13weeks
# from interest_rates import Tresury_bond_5years
# from interest_rates import Tresury_bond_30years

if __name__ == '__main__':

    t = 0
    K = 100
    vol = 0.6
    r = 0.1
    T = 5/365.6
    T2 = 100/365.6

    S0 = 100
    strike = 100

    stock1 = asset_BS(2.27, 0)
    callEU = Option_eu(100, 'Call EU', stock1, 2.3, T, r, vol)
    callEU2 = Option_eu(-1, 'Put EU', stock1, strike, T2, r, vol)
    callEU_Barrier = Option_eu(1, 'Call In & Out', stock1, strike, T, r, vol, 200)
    # callEU_Barrier.option_price_mc()
    book1 = Book([callEU])

    call_spread = Option_prem_gen(-1, 'Call Spread', stock1, [95, 105], T, r, vol)
    straddle1 = Option_prem_gen(1, 'Strangle', stock1, [95, 95], T, r, vol)
    straddle2 = Option_prem_gen(-1, 'Strangle', stock1, [95, 95],  T2, r, vol)

    book1 = Book([callEU])
    book2 = Book([straddle1, straddle2])
    book3 = Book([callEU, callEU2])

    book2.simu_asset(time=5)

    book1.delta_hedge()
    book1.simu_asset(time=1)
    book1.Delta_DF()

    book1.simu_asset(time=5)
    book1.pnl()


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