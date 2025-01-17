import numpy as np
from matplotlib import pyplot as plt

from Graphics.Graphics import plot_2d, plotPnl
from Options.Options_class import  Option_eu, Option_prem_gen
import copy
import plotly.graph_objects as go
from Logger.Logger import mylogger
class Book(Option_eu):
    def __init__(self, options_basket:list, name:str=None, logger=False)->None:
        self.name = name
        self.basket = options_basket
        self.asset = self.basket[0].asset
        # self.asset = list(set(option.asset for option in self.basket)) multi assets book - may not be a nice idea
        self.book_old = None
        if logger:
            mylogger.logger.info(f"Book has been intiated : Basket of options= {options_basket}")
        return
    def append(self, option:(Option_eu, Option_prem_gen))->None:
        self.basket.append(option)
        return

    def delta_hedge(self, logger=False):
        if logger:
            mylogger.logger.info('Start delta hedging.')
        unique_asset = list(set([option.asset for option in self.basket]))
        #considering 1 unique asset
        unique_asset = unique_asset[0]
        delta = round(-self.Delta_DF())
        unique_asset.quantity = delta
        if logger:
            mylogger.logger.info('Delta hedging done. SUCCESS.')
        return

    def get_move_deltahedge(self):
        st = self.asset.St
        move = []
        while self.Delta_DF() < 1:
            self.asset.St += 0.01
        move.append(self.asset.St)
        while self.Delta_DF() > -1:
            self.asset.St += -0.01
        move.append(self.asset.St)
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

    def Delta_DF(self):
        hedge = self.asset.quantity if self.asset != None else 0
        return sum([option.Delta_DF() for option in self.basket]) + hedge
    def Delta_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 0.5)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = [option.T for option in self.basket]
        list_delta = []

        range_t = [t / (365*100) for t in range(0, round(min(option_matu) * 100*365), 2)]

        for t in range_t:
            for option in self.basket:
                option.T = min(option_matu) - t
            for st in range_st:
                self.asset.St = st
                list_delta.append(self.Delta_DF())

        self.asset.St = asset_st
        for option, matu in zip(self.basket, option_matu):
            option.T=matu
        range_st_mesh, range_t_mesh = np.meshgrid(range_st, range_t)
        list_delta = np.array(list_delta).reshape(len(range_t), len(range_st))

        fig = go.Figure(data=[go.Surface(z=list_delta, x=range_st_mesh, y=range_t_mesh * 365, colorscale='magma')])
        fig.update_layout(title='Delta of the Option vs. Underlying Asset Price and Time',
                          scene=dict(
                              xaxis_title='Underlying Asset Price (St)',
                              yaxis_title='Time to Maturity (T, days)',
                              zaxis_title='Option Delta'
                          ))
        fig.show()
    def Gamma_DF(self):
        return sum([option.Gamma_DF() for option in self.basket])
    def Gamma_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = [option.T for option in self.basket]
        list_gamma = []

        range_t = [t / (365*100) for t in range(0, round(min(option_matu) * 100*365), 2)]

        for t in range_t:
            for option in self.basket:
                option.T = min(option_matu) - t
            for st in range_st:
                self.asset.St = st
                list_gamma.append(self.Gamma_DF())

        self.asset.St = asset_st
        for option, matu in zip(self.basket, option_matu):
            option.T=matu
        range_st_mesh, range_t_mesh = np.meshgrid(range_st, range_t)
        list_gamma = np.array(list_gamma).reshape(len(range_t), len(range_st))
        fig = go.Figure(data=[go.Surface(z=list_gamma, x=range_st_mesh, y=range_t_mesh * 365, colorscale='magma')])
        fig.update_layout(title='Gamma of the Option vs. Underlying Asset Price and Time',
                          scene=dict(
                              xaxis_title='Underlying Asset Price (St)',
                              yaxis_title='Time to Maturity (T, days)',
                              zaxis_title='Option Gamma'
                          ))
        fig.show()
    def Speed_DF(self):
        return sum([option.Speed_DF() for option in self.basket])
    def Vega_DF(self):
        return sum([option.Vega_DF() for option in self.basket])
    def Vega_surface(self):
        vol_option = max([option.sigma for option in self.basket])
        option_matu = [option.T for option in self.basket]
        list_vega = []
        range_sigma = [sigma / 100 for sigma in range(round(vol_option * 100*0.5), round(vol_option * 100*1.5), 2)]
        range_t = [t / (365*100) for t in range(0, round(min(option_matu) * 100*365), 2)]

        for t in range_t:
            for option in self.basket:
                option.T = min(option_matu) - t
            for sigma_ in range_sigma:
                self.sigma = sigma_
                list_vega.append(self.Vega_DF())

        self.sigma = vol_option
        for option, matu in zip(self.basket, option_matu):
            option.T=matu
        range_sigma_mesh, range_t_mesh = np.meshgrid(range_sigma, range_t)
        list_vega = np.array(list_vega).reshape(len(range_t), len(range_sigma))
        fig = go.Figure(data=[go.Surface(z=list_vega, x=range_sigma_mesh, y=range_t_mesh * 365, colorscale='magma')])
        fig.update_layout(title='Vega of the Option vs. Underlying Asset Price and Time',
                          scene=dict(
                              xaxis_title='Underlying Asset Price (St)',
                              yaxis_title='Time to Maturity (T, days)',
                              zaxis_title='Option Vega'
                          ))
        fig.show()
    def Theta_DF(self):
        return sum([option.Theta_DF() for option in self.basket])
    def Theta_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = [option.T for option in self.basket]
        list_theta = []

        range_t = [t / (365*100) for t in range(0, round(min(option_matu) * 100*365), 2)]

        for t in range_t:
            for option in self.basket:
                option.T = min(option_matu) - t
            for st in range_st:
                self.asset.St = st
                list_theta.append(self.Theta_DF())

        self.asset.St = asset_st
        for option, matu in zip(self.basket, option_matu):
            option.T=matu
        range_st_mesh, range_t_mesh = np.meshgrid(range_st, range_t)
        list_theta = np.array(list_theta).reshape(len(range_t), len(range_st))
        fig = go.Figure(data=[go.Surface(z=list_theta, x=range_st_mesh, y=range_t_mesh * 365, colorscale='magma')])
        fig.update_layout(title='Theta of the Option vs. Underlying Asset Price and Time',
                          scene=dict(
                              xaxis_title='Underlying Asset Price (St)',
                              yaxis_title='Time to Maturity (T, days)',
                              zaxis_title='Option Theta'
                          ))
        fig.show()

    def Vanna_DF(self):
        return sum([option.Vanna_DF() for option in self.basket])

    def Vanna_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = [option.T for option in self.basket]
        list_vanna = []

        range_t = [t / (365 * 100) for t in range(0, round(min(option_matu) * 100 * 365), 2)]

        for t in range_t:
            for option in self.basket:
                option.T = min(option_matu) - t
            for st in range_st:
                self.asset.St = st
                list_vanna.append(self.Vanna_DF())

        self.asset.St = asset_st
        for option, matu in zip(self.basket, option_matu):
            option.T = matu
        range_st_mesh, range_t_mesh = np.meshgrid(range_st, range_t)
        list_vanna = np.array(list_vanna).reshape(len(range_t), len(range_st))
        fig = go.Figure(data=[go.Surface(z=list_vanna, x=range_st_mesh, y=range_t_mesh * 365, colorscale='magma')])
        fig.update_layout(title='Vanna of the Option vs. Underlying Asset Price and Time',
                          scene=dict(
                              xaxis_title='Underlying Asset Price (St)',
                              yaxis_title='Time to Maturity (T, days)',
                              zaxis_title='Option Vanna'
                          ))
        fig.show()

    def Volga_DF(self):
        return sum([option.Volga_DF() for option in self.basket])

    def display_payoff_option(self, plot=True):
        Purchase = sum([x.option_price_close_formulae() for x in self.basket])
        St_init = self.asset.St
        if self.asset.St > 10:
            St = range(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 1)
        else:
            St = [x / 100 for x in range(round(self.asset.St * 0.5 * 100), round(self.asset.St * 3 * 100), 1)]
        payoffs = [sum(x) - Purchase for x in zip(*[option.display_payoff_option(plot=False, range=St)[1] for option in self.basket])]
        asset_contributions = [(st - St_init) * self.asset.quantity for st in St] if self.asset.quantity != 0 else [0] * len(St)
        book_payoff = [sum(x) for x in zip(*[payoffs, asset_contributions])]
        plt.clf()
        plot_2d(St, book_payoff, "Asset price", "Payoff", plot=plot, title=f"Book payoff")
        return
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
    def Third_Order_Pnl(self, plot=True):#Pnl due to Gamma-convexity and Vomma Effect
        tableau_pnl = sum([option.Third_Order_Pnl(plot=False) for option in self.basket])
        if plot:
            plotPnl(list(tableau_pnl), tableau_pnl.index, '3rd ORDER PNL')
        return tableau_pnl
