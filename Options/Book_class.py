from Graphics.Graphics import plot_2d
from Options.Options_class import  Option_eu, Option_prem_gen
import copy
class Book(Option_eu):
    def __init__(self, options_basket:list)->None:
        self.basket = options_basket
        self.asset = self.basket[0].asset
        # self.asset = list(set(option.asset for option in self.basket)) multi assets book - may not be a nice idea
        self.book_old = None
        return
    def append(self, option:(Option_eu, Option_prem_gen))->None:
        self.basket.append(option)
        return

    def delta_hedge(self):
        unique_asset = list(set([option.asset for option in self.basket]))
        #considering 1 unique asset
        unique_asset = unique_asset[0]
        delta = round(-self.Delta_DF())
        unique_asset.quantity = delta
        return

    def get_move_deltahedge(self):
        st = self.asset.St
        while self.Delta_DF() < 1:
            self.asset.St += 0.01
        move = self.asset.St
        self.asset.St = st
        return move

    def option_price_close_formulae(self):
        return sum([option.option_price_close_formulae() if isinstance(option, (Option_eu, Option_prem_gen)) else 0 for option in self.basket]) + self.asset.quantity*self.asset.St
    def get_payoff_option(self, ST:int):#to correct
        payoff = 0
        for option in self.basket:
            payoff+=option.get_payoff_option(ST)
        if self.asset:
            payoff += ST*self.asset.quantity
        return payoff
    def display_payoff_option(self):
        K_min = 99999999
        K_max = -99999999

        for option in self.basket:
            if type(option.K) == list:
                K_min_temp = min(option.K)
                K_max_temp = max(option.K)
            else:
                K_min_temp = option.K
                K_max_temp = option.K

            if K_min_temp < K_min:
                K_min = K_min_temp
            if K_max_temp > K_max:
                K_max = K_max_temp
        start = K_min*0.5
        end = K_max*1.5
        ST = list(range(round(start), round(end)))
        payoffs = []
        for i in ST:
            payoffs.append(self.get_payoff_option(i))
        plot_2d(ST, payoffs, "Asset price", "Payoff", isShow=True, title="Payoff of the book")
    def Delta_DF(self):
        hedge = self.asset.quantity if self.asset != None else 0
        return sum([option.Delta_DF() for option in self.basket]) + hedge
    def Gamma_DF(self):
        return sum([option.Gamma_DF() for option in self.basket])
    def Vega_DF(self):
        return sum([option.Vega_DF() for option in self.basket])
    def Theta_DF(self):
        return sum([option.Theta_DF() for option in self.basket])
    def simu_asset(self, time):
        self.book_old = copy.deepcopy(self)
        list_asset = list(set([x.asset for x in self.basket]))
        for item in list_asset:
            item.simu_asset(time)
        for option in self.basket:
            option.update_t(time)
        return
    def clean_basket(self):

        for i in self.basket:
            for h in self.basket:
                if h == i:
                    continue
                elif i.type == h.type and i.asset == h.asset and i.T == h.T and i.K == h.K:
                    i.position += h.position
                    h.position = 0
                    self.basket.remove(h)
        return
    def pnl(self):
        delta_pnl = self.Delta_DF() - self.book_old.Delta_DF()
        gamma_pnl = self.Gamma_DF() - self.book_old.Gamma_DF()
        theta_pnl = self.Theta_DF() - self.book_old.Theta_DF()
        vega_pnl = self.Vega_DF() - self.book_old.Vega_DF()
        asset_price_delta = self.basket[0].asset.St - self.book_old.basket[0].asset.St
        explained_pnl = delta_pnl*asset_price_delta + 0.5*gamma_pnl*asset_price_delta**2 + theta_pnl*asset_price_delta + vega_pnl*asset_price_delta
        return explained_pnl