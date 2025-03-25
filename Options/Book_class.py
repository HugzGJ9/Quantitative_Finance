import numpy as np
from matplotlib import pyplot as plt

from Graphics.Graphics import plot_2d, plotPnl, plotGreek
from Options.Options_class import  Option_eu, Option_prem_gen
import copy
import plotly.graph_objects as go
from Logger.Logger import mylogger
class Book():
    def __init__(self, options_basket:list, name:str=None, logger=False)->None:
        self.name = name
        self.basket = options_basket
        self.asset = self.basket[0].asset if self.basket else None
        # self.asset = list(set(option.asset for option in self.basket)) multi assets book - may not be a nice idea
        self.book_old = None
        if logger:
            mylogger.logger.info(f"Book has been intiated : Basket of options= {options_basket}")
        return

    def append(self, option: (Option_eu, Option_prem_gen)) -> None:
        if self.asset and option.asset != self.asset:
            raise ValueError(f"Cannot add option with asset {option.asset} to book of {self.asset}")
        self.basket.append(option)

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
    def get_payoff_option(self, ST:int):
        payoff = 0
        for option in self.basket:
            payoff+=option.get_payoff_option(ST)
        return payoff
    def GreekRisk(self, greek, logger=False, plot=True):
        greek_map = {
            "Delta": (self.Delta_DF, Option_eu.DeltaRisk),
            "Gamma": (self.Gamma_DF, Option_eu.GammaRisk),
            "Theta": (self.Theta_DF, Option_eu.ThetaRisk),
            "Vega": (self.Vega_DF, Option_eu.VegaRisk),
            "Vanna": (self.Vanna_DF, Option_eu.VannaRisk),
            "Volga": (self.Volga_DF, Option_eu.VolgaRisk),
            "Speed": (self.Speed_DF, Option_eu.SpeedRisk),
        }
        if greek not in greek_map:
            raise ValueError(f"Invalid Greek specified: {greek}")
        GreekDF, GreekRisk = greek_map[greek]
        hedge = self.asset.quantity if self.asset != None else 0
        Greek_Book = GreekDF() + hedge
        GreekRisk_df = sum([GreekRisk(option, plot=False) + hedge for option in self.basket])
        if plot:
            plotGreek(self.asset.St, Greek_Book, GreekRisk_df['value'], GreekRisk_df.index, greek)
        return GreekRisk_df
    def Delta_DF(self):
        hedge = self.asset.quantity if self.asset != None else 0
        return sum([option.Delta_DF() for option in self.basket]) + hedge
    def DeltaRisk(self, logger=False, plot=True):
        greek = "Delta"
        DeltaRisk_df = self.GreekRisk(greek, logger, plot)
        return DeltaRisk_df
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

    def GammaRisk(self, logger=False, plot=True):
        greek = "Gamma"
        GammaRisk_df = self.GreekRisk(greek, logger, plot)
        return GammaRisk_df
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
    def SpeedRisk(self, logger=False, plot=True):
        greek = "Speed"
        SpeedRisk_df = self.GreekRisk(greek, logger, plot)
        return SpeedRisk_df
    def Vega_DF(self):
        return sum([option.Vega_DF() for option in self.basket])
    def VegaRisk(self, logger=False, plot=True):
        greek = "Vega"
        VegaRisk_df = self.GreekRisk(greek, logger, plot)
        return VegaRisk_df
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
    def ThetaRisk(self, logger=False, plot=True):
        greek = "Theta"
        ThetaRisk_df = self.GreekRisk(greek, logger, plot)
        return ThetaRisk_df
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
    def VannaRisk(self, logger=False, plot=True):
        greek = "Vanna"
        VannaRisk_df = self.GreekRisk(greek, logger, plot)
        return VannaRisk_df
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
    def VolgaRisk(self, logger=False, plot=True):
        greek = "Volga"
        VolgaRisk_df = self.GreekRisk(greek, logger, plot)
        return VolgaRisk_df

    def RiskAnalysis(self, logger=False):
        if logger:
            mylogger.logger.info("Start Risk Analysis.")
        self.DeltaRisk(logger=logger)
        self.GammaRisk(logger=logger)
        self.VegaRisk(logger=logger)
        self.ThetaRisk(logger=logger)
        self.VannaRisk(logger=logger)
        self.VolgaRisk(logger=logger)
        self.SpeedRisk(logger=logger)
        if logger:
            mylogger.logger.info("Risk Analysis done. SUCCESS.")
        return
    def display_payoff_option(self, plot=True):
        St_init = self.asset.St
        if self.asset.St > 10:
            ST = list(range(round(self.asset.St * 0.8), round(self.asset.St * 1.2), 1))
        else:
            ST = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 1)]
        payoffs = [sum(x) for x in zip(*[option.display_payoff_option(plot=False, asset_range=ST)[1] for option in self.basket])]
        asset_contributions = [(st - St_init) * self.asset.quantity for st in ST] if self.asset.quantity != 0 else [0] * len(ST)
        book_payoff = [sum(x) for x in zip(*[payoffs, asset_contributions])]
        if plot:
            plt.clf()
        plot_2d(ST, book_payoff, "Asset price", "Payoff", plot=plot, title=f"Book payoff")
        return [ST, payoffs]
    def simu_asset(self, time):
        self.book_old = {option: option.T for option in self.basket}
        list_asset = list(set([x.asset for x in self.basket]))
        for item in list_asset:
            item.simu_asset(time)
        for option in self.basket:
            option.update_t(time)
        return
    def clean_basket(self):
        self.basket = [
            i for i in self.basket if i.position != 0
        ]
    def PnlRisk(self, plot=True):
        hedge = self.asset.quantity if self.asset != None else 0
        PnlRisk_df = sum([option.PnlRisk(plot=False) + hedge for option in self.basket])
        if plot:
            plotGreek(self.asset.St, 0, PnlRisk_df['value'], PnlRisk_df.index, "Pnl")
        return PnlRisk_df
    def Delta_Pnl(self, plot=True):
        PnlRisk_df = sum([option.Delta_Pnl(plot=False) for option in self.basket])
        if plot:
            plotPnl(list(PnlRisk_df['value']), PnlRisk_df.index, 'DELTA PNL')
        return PnlRisk_df
    def Gamma_Pnl(self, plot=True):
        PnlRisk_df = sum([option.Gamma_Pnl(plot=False) for option in self.basket])
        if plot:
            plotPnl(list(PnlRisk_df['value']), PnlRisk_df.index, 'GAMMA PNL')
        return PnlRisk_df
    def Third_Order_Pnl(self, plot=True):#Pnl due to Gamma-convexity and Vomma Effect
        PnlRisk_df = sum([option.Third_Order_Pnl(plot=False) for option in self.basket])
        if plot:
            plotPnl(list(PnlRisk_df['value']), PnlRisk_df.index, '3rd ORDER PNL')
        return PnlRisk_df
    def nOrderPnl(self, plot=True):
        tableau_pnl = sum(
            pnl_func(plot=False) for pnl_func in [self.Delta_Pnl, self.Gamma_Pnl, self.Third_Order_Pnl]
        )
        if plot:
            plotPnl(list(tableau_pnl['value']), tableau_pnl.index, 'N ORDER PNL')
        return tableau_pnl