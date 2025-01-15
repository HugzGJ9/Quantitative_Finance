import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from Asset_Modeling.Asset_class import asset_BS
from  Asset_Modeling.Actif_stoch_BS import simu_actif
from Graphics.Graphics import plot_2d, plotGreek, plotPnl
from Options.payoffs import payoff_call_eu, payoff_put_eu, payoff_call_asian, payoff_put_asian, close_formulae_call_eu, \
    close_formulae_put_eu, payoff_call_eu_barrier, close_formulae_magrabe
import plotly.graph_objects as go
from Logger.Logger import mylogger
from scipy.interpolate import RegularGridInterpolator
import os


class Option_eu:
    #root parameter to
    def __init__(self, position:int, type:str, asset:(asset_BS), K:float, T:float, r:float, sigma=None, volatility_surface_df=None, use_vol_surface=False, barrier=None, logger= False, booked_price=None):
        self.position = position
        self.asset = asset
        self.type = type
        self.K = K
        self.t = (len(self.asset.history)-1)/(365*24)
        self.T = T
        self.r = r
        self.barrier = barrier
        self.sigma = sigma
        self.volatility_surface_df = volatility_surface_df
        self.use_vol_surface = use_vol_surface
        self.volatility_interpolator = None
        self.booked_price = booked_price

        if self.use_vol_surface and self.volatility_surface_df is not None:
            self.set_volatility_surface()

        if logger:
            mylogger.logger.info(
                f"Option initialized: {type=} - {position=} - maturity={T * 365}days - strike={K}"
            )

    def set_volatility_surface(self):
        self.volatility_surface_df['Strike_percentage'] = self.volatility_surface_df['Strike_percentage'].replace("ATM", 0).astype(
            float)
        self.strike = self.volatility_surface_df['Strike_percentage'].to_numpy()

        self.maturities = np.array(
            [1 / 365, 7 / 365, 30 / 365, 90 / 365, 180 / 365, 365 / 365])
        self.vol_data = self.vol_data = self.volatility_surface_df.iloc[:, 1:].to_numpy()
        self.volatility_interpolator = RegularGridInterpolator(
            (self.strike, self.maturities), self.vol_data, bounds_error=False, fill_value=None
        )
    def show_volatility_surface(self):
        columns = self.volatility_surface_df.columns
        for i in range(1, len(columns)):
            plot_2d(self.volatility_surface_df['Strike_percentage'], self.volatility_surface_df[columns[i]], 'STRIKE %', 'Volatility', plot=False, legend=list(columns[1:]))
        plt.plot(self.K / self.asset.St - 1, self.get_sigma(), 'x', label=f'Option sigma',
                 color='red', markersize=10, markeredgewidth=2)
        formatter = FuncFormatter(lambda y, _: f'{y * 100:.0f}%')
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()

    def show_volatility_stability(self):
        vol = []
        ST = self.asset.St
        for st in range(int(ST*0.8 * 1000), int(ST*1.2 * 1000), 1):
            self.asset.St = st/1000
            vol.append(self.get_sigma())
        self.asset.St = ST
        plt.plot(vol)

    def get_sigma(self):
        """Fetch volatility dynamically from the surface or use constant sigma."""
        if self.use_vol_surface:
            self.sigma = round(float(self.volatility_interpolator((self.K / self.asset.St - 1, self.T))), 4)
            return self.sigma
        elif self.sigma is not None:
            return self.sigma
        else:
            raise ValueError("Either constant sigma or volatility surface must be defined.")

    def option_price_close_formulae(self):
        sigma = self.get_sigma()
        if self.type == "Call EU":
            option_price = close_formulae_call_eu(self.asset.St, self.K, self.t, self.T, self.r, sigma)
            return self.position * option_price
        elif self.type == "Put EU":
            option_price = close_formulae_put_eu(self.asset.St, self.K, self.t, self.T, self.r, sigma)
            return self.position * option_price
    def update_t(self, days):
        self.t = days/365
        return
    def get_payoff_option(self, ST:int)->int:
        Purchase = self.option_price_close_formulae()
        if self.type == "Call EU":
            payoff = payoff_call_eu(ST, self.K) * self.position - Purchase
        elif self.type == "Put EU":
            payoff = payoff_put_eu(ST, self.K) * self.position - Purchase
        return payoff
    def display_payoff_option(self, plot=True, asset_range=None):
        if asset_range is not None:
            ST = asset_range
        else:
            if self.K>10:
                ST = list(range(round(self.K*0.5), round(self.K*1.5), 2))
            else:
                ST = [x/100 for x in range(round(self.K*0.5*100), round(self.K*3*100), 2)]

        payoffs = []
        for i in ST:
            payoffs.append(self.get_payoff_option(i))
        plot_2d(ST, payoffs, "Asset price", "Payoff", plot=plot, title=f"{self.type} payoff")
        return [ST, payoffs]
    def option_price_mc(self, Nmc=1000):
        prix_option = 0
        for i in range(Nmc):
            prix_actif = simu_actif(self.asset.St, self.t, self.T, self.r, self.get_sigma())
            if self.type == "Call EU":
                prix_option += payoff_call_eu(prix_actif[-1], self.K)
            elif self.type == "Put EU":
                prix_option += payoff_put_eu(prix_actif[-1], self.K)
            elif self.type == "Call Asian":
                prix_option += payoff_call_asian(prix_actif, self.K)
            elif self.type == "Put Asian":
                prix_option += payoff_put_asian(prix_actif, self.K)
            elif self.type == "Call Up & Out":
                activation_barrier = True in [self.barrier<St for St in prix_actif]
                if activation_barrier:
                    print('barrier activated')
                    plt.plot(prix_actif, color='red')
                else:
                    plt.plot(prix_actif, color='green')
                prix_option += payoff_call_eu_barrier(prix_actif[-1], self.K, activation_barrier)
        prix_option = np.exp(-self.r*(self.T-self.t))*prix_option / Nmc

        return self.position*prix_option

    def option_price_binomial_tree(self, daily=True):
        """
        Price an option using the binomial tree model.

        Parameters:
        daily (bool): If True, use daily steps. If False, use annual steps.

        Returns:
        float: Option price at the root of the binomial tree.
        """
        steps_per_year = 365 if daily else 1
        volatility = self.sigma / np.sqrt(steps_per_year) if daily else self.sigma

        up = 1 + volatility
        down = 1 - volatility
        discount_factor = np.exp(-self.r / steps_per_year)
        q = (np.exp(self.r / steps_per_year) - down) / (up - down)

        time_to_expiry = int((self.T - self.t) * steps_per_year)

        asset_list = [[self.asset.St]]

        for _ in range(time_to_expiry):
            previous_prices = asset_list[-1]
            new_prices = [price * up for price in previous_prices] + [price * down for price in previous_prices]
            asset_list.append(new_prices)

        if 'Call' in self.type:
            option_values = [max(price - self.K, 0) for price in asset_list[-1]]
        elif 'Put' in self.type:
            option_values = [max(self.K - price, 0) for price in asset_list[-1]]

        for n in range(time_to_expiry - 1, -1, -1):
            option_values = [
                discount_factor * (q * option_values[i] + (1 - q) * option_values[i + 1])
                for i in range(0, len(option_values), 2)
            ]

        return option_values[0]

    def Delta_DF(self):
        delta_St = 0.001
        asset = asset_BS(self.asset.St, self.asset.quantity, self.asset.name)
        self.get_sigma()
        option_replica = Option_eu(self.position, self.type, asset, self.K, self.T, self.r, self.sigma)

        asset.St = asset.St + delta_St
        option_delta_St = option_replica.option_price_close_formulae()
        asset.St = asset.St - 2*delta_St
        option_option = option_replica.option_price_close_formulae()
        asset.St = asset.St + delta_St
        delta = (option_delta_St - option_option)/(2*delta_St)
        return delta

    def DeltaRisk(self, logger=False, plot=True):
        if logger:
            mylogger.logger.info("Start Delta Risk.")
        Delta_Option, list_delta, range_st = self.computeGreekCurve("Delta")
        if plot:
            plotGreek(self.asset.St, Delta_Option, list_delta, range_st, 'Delta')
        tableau_delta = pd.DataFrame(zip(range_st, list_delta), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_delta.index = tableau_delta['Underlying Asset Price (St) move']
        tableau_delta = tableau_delta['gains']
        if logger:
            mylogger.logger.info("Delta Risk done. SUCCESS.")
        return tableau_delta

    def computeGreekCurve(self, greek):
        if greek == "Delta":
            Greek_DF = self.Delta_DF
        elif greek == "Gamma":
            Greek_DF = self.Gamma_DF
        elif greek == "Vega":
            Greek_DF = self.Vega_DF
        elif greek == "Theta":
            Greek_DF = self.Theta_DF
        elif greek == "Vanna":
            Greek_DF = self.Vanna_DF
        elif greek == "Volga":
            Greek_DF = self.Volga_DF
        elif greek == "Skew":
            Greek_DF = self.Skew_DF
        elif greek == "Speed":
            Greek_DF = self.Speed_DF
        else:
            raise ValueError(f"Invalid Greek specified: {greek}")
        Greek_Option = Greek_DF()
        if self.asset.St > 10:
            range_st = range(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 1)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 1)]
        asset_st = self.asset.St
        list_greek = list()
        for st in range_st:
            self.asset.St = st
            list_greek.append(Greek_DF())
        self.asset.St = asset_st
        return Greek_Option, list_greek, range_st

    def Delta_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 0.5)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = self.T
        list_delta = []

        range_t = [t / (365*100) for t in range(0, round(self.T * 100*365), 2)]

        for t in range_t:
            self.T = option_matu - t
            for st in range_st:
                self.asset.St = st
                list_delta.append(self.Delta_DF())

        self.asset.St = asset_st
        self.T = option_matu

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
        delta_St = 0.001
        asset = asset_BS(self.asset.St, self.asset.quantity, self.asset.name)
        self.get_sigma()
        option_replica = Option_eu(self.position, self.type, asset, self.K, self.T, self.r, self.sigma)

        asset.St = asset.St + delta_St
        option_delta_St = option_replica.Delta_DF()
        asset.St = asset.St - 2 * delta_St
        option_option = option_replica.Delta_DF()
        asset.St = asset.St + delta_St
        gamma = (option_delta_St - option_option) / (2 * delta_St)
        return gamma

    def GammaRisk(self, logger=False, plot=True):
        if logger:
            mylogger.logger.info("Start Gamma Risk.")
        Gamma_Option, list_gamma, range_st = self.computeGreekCurve('Gamma')
        if plot:
            plotGreek(self.asset.St, Gamma_Option, list_gamma, range_st, 'Gamma')
        tableau_gamma = pd.DataFrame(zip(range_st, list_gamma), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_gamma.index = tableau_gamma['Underlying Asset Price (St) move']
        tableau_gamma = tableau_gamma['gains']
        if logger:
            mylogger.logger.info("Gamma Risk done. SUCCESS.")
        return tableau_gamma

    def Gamma_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = self.T
        list_gamma = []

        range_t = [t / (365*100) for t in range(0, round(self.T * 100*365), 2)]

        for t in range_t:
            self.T = option_matu - t
            for st in range_st:
                self.asset.St = st
                list_gamma.append(self.Gamma_DF())

        self.asset.St = asset_st
        self.T = option_matu

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
        delta_St = 0.001
        asset = asset_BS(self.asset.St, self.asset.quantity, self.asset.name)
        self.get_sigma()
        option_replica = Option_eu(self.position, self.type, asset, self.K, self.T, self.r, self.sigma)

        asset.St = asset.St + delta_St
        option_delta_St = option_replica.Gamma_DF()
        asset.St = asset.St - 2 * delta_St
        option_option = option_replica.Gamma_DF()
        asset.St = asset.St + delta_St
        speed = (option_delta_St - option_option) / (2 * delta_St)
        return speed

    def SpeedRisk(self, logger=False, plot=True):
        if logger:
            mylogger.logger.info("Start Speed Risk.")
        Speed_Option, list_speed, range_st = self.computeGreekCurve("Speed")
        if plot:
            plotGreek(self.asset.St, Speed_Option, list_speed, range_st, 'Speed')
        tableau_speed = pd.DataFrame(zip(range_st, list_speed), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_speed.index = tableau_speed['Underlying Asset Price (St) move']
        tableau_speed = tableau_speed['gains']
        if logger:
            mylogger.logger.info("Speed Risk done. SUCCESS.")
        return tableau_speed
    def Vega_DF(self):
        sigma = self.get_sigma()
        delta_vol = 0.00001

        option_delta_vol = Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r,
                                    sigma+delta_vol).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r,
                                  sigma-delta_vol).option_price_close_formulae()

        vega = (option_delta_vol - option_option) / (2*delta_vol)
        return vega/100

    def VegaRisk(self, logger=False, plot=True):
        if logger:
            mylogger.logger.info("Start Vega Risk.")
        Vega_Option, list_vega, range_st = self.computeGreekCurve("Vega")
        if plot:
            plotGreek(self.asset.St, Vega_Option, list_vega, range_st, 'Vega')
        tableau_vega = pd.DataFrame(zip(range_st, list_vega), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_vega.index = tableau_vega['Underlying Asset Price (St) move']
        tableau_vega = tableau_vega['gains']
        if logger:
            mylogger.logger.info("Vega Risk done. SUCCESS.")
        return tableau_vega
    def Vega_surface(self):
        option_matu = self.T
        list_vega = []
        sigma = self.get_sigma()
        range_sigma = [sigma / 100 for sigma in range(round(sigma * 100*0.5), round(sigma * 100*1.5), 2)]
        range_t = [t / (365*100) for t in range(0, round(self.T * 100*365), 2)]

        for t in range_t:
            self.T = option_matu - t
            for sigma_ in range_sigma:
                list_vega.append(Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r,
                                  sigma_).Vega_DF())
        self.T = option_matu

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
        delta_t = 0.00001
        self.t = self.t+delta_t
        option_delta_t = self.option_price_close_formulae()
        self.t = self.t-delta_t
        option_option = self.option_price_close_formulae()
        theta = (option_delta_t - option_option) / delta_t
        return theta/365

    def ThetaRisk(self, logger=False, plot=True):
        if logger:
            mylogger.logger.info("Start Theta Risk.")
        Theta_Option, list_theta, range_st = self.computeGreekCurve("Theta")
        if plot:
            plotGreek(self.asset.St, Theta_Option, list_theta, range_st, 'Theta')
        tableau_theta = pd.DataFrame(zip(range_st, list_theta), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_theta.index = tableau_theta['Underlying Asset Price (St) move']
        tableau_theta = tableau_theta['gains']
        if logger:
            mylogger.logger.info("Theta Risk done. SUCCESS.")
        return tableau_theta

    def Theta_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = self.T
        list_theta = []

        range_t = [t / (365*100) for t in range(0, round(self.T * 100*365), 2)]

        for t in range_t:
            self.T = option_matu - t
            for st in range_st:
                self.asset.St = st
                list_theta.append(self.Theta_DF())

        self.asset.St = asset_st
        self.T = option_matu

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
    def Volga_DF(self): #Recall : Volga corresponds to the Vega change regarding the IV change, it is the Vega convexity
        delta_vol = 0.001
        sigma = self.get_sigma()
        option_delta_vol = Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r,
                                     sigma + delta_vol).Vega_DF()
        option_option = Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r,
                                  sigma).Vega_DF()
        volga = (option_delta_vol - option_option) / delta_vol
        return volga

    def VolgaRisk(self, logger=False, plot=True):
        if logger:
            mylogger.logger.info("Start Volga Risk.")
        Volga_Option, list_volga, range_st = self.computeGreekCurve("Volga")
        if plot:
            plotGreek(self.asset.St, Volga_Option, list_volga, range_st, 'Volga')
        tableau_volga = pd.DataFrame(zip(range_st, list_volga), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_volga.index = tableau_volga['Underlying Asset Price (St) move']
        tableau_volga = tableau_volga['gains']
        if logger:
            mylogger.logger.info("Volga Risk done. SUCCESS.")
        return tableau_volga
    def Vanna_DF(self):
        delta_St = 0.00001
        self.asset.St = self.asset.St + delta_St
        option_delta_vol = self.Vega_DF()
        self.asset.St = self.asset.St - delta_St
        option_option = self.Vega_DF()
        vanna = (option_delta_vol - option_option) / delta_St
        return vanna

    def VannaRisk(self, logger=False, plot=True):
        if logger:
            mylogger.logger.info("Start Vanna Risk.")
        Vanna_Option, list_vanna, range_st = self.computeGreekCurve("Vanna")
        if plot:
            plotGreek(self.asset.St, Vanna_Option, list_vanna, range_st, 'Vanna')
        tableau_vanna = pd.DataFrame(zip(range_st, list_vanna), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_vanna.index = tableau_vanna['Underlying Asset Price (St) move']
        tableau_vanna = tableau_vanna['gains']
        if logger:
            mylogger.logger.info("Vanna Risk done. SUCCESS.")
        return tableau_vanna

    def Skew_DF(self):
        St = self.asset.St
        sigma = self.get_sigma()
        self.asset.St = self.K
        sigma_ATM = self.get_sigma()
        self.asset.St = St
        return sigma - sigma_ATM
    def SkewRisk(self, logger=False):
        if logger:
            mylogger.logger.info("Start Skew Risk.")
        Skew_Option, list_skew, range_st = self.computeGreekCurve("Skew")
        plotGreek(self.asset.St, Skew_Option, list_skew, range_st, 'Skew')
        if logger:
            mylogger.logger.info("Skew Risk done. SUCCESS.")
        return
    def Vega_convexity(self, plot=True):
        Vega_Option = self.Vega_DF()
        range_vol = [x / 100 for x in range(1, 100)]
        sigma = self.get_sigma()
        list_vega = list()
        for vol in range_vol:
            list_vega.append(Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r, vol).Vega_DF())
        plt.plot(range_vol, list_vega, label='Vega vs Implied Volatility (IV)', color='blue', linestyle='-', linewidth=2)
        plt.plot(sigma, Vega_Option, 'x', label='Option Vega at Specific Points',
                 color='red', markersize=10, markeredgewidth=2)

        # Adding titles and labels
        plt.title('Vega of the Option vs. Implied Volatility (IV)')
        plt.xlabel('Implied Volatility (IV)')
        plt.ylabel('Option Vega')
        plt.legend()
        plt.grid(True)
        if plot:
            plt.show()
        return
    def simu_asset(self, time):
        self.asset.simu_asset(time)
        #self.asset.St = self.asset.history[-1]
        self.t = self.t + time/365

    def PnlRisk(self, plot=True):
        Price_Option = self.option_price_close_formulae()
        # range_st = np.arange(self.asset.St-3, self.asset.St+3, 0.1)

        if self.asset.St>1000:
            range_st = range(round(self.asset.St*0.9), round(self.asset.St*1.1), 1)
        elif self.asset.St>10:
            range_st = range(round(self.asset.St*0.9), round(self.asset.St*1.1), 1)
        else:
            range_st = [x/100 for x in range(round((self.asset.St - 1 ) *100), round((self.asset.St + 1 ) *100), 1)]

        asset_st = self.asset.St
        list_price = list()
        for st in range_st:
            self.asset.St = st
            list_price.append(self.option_price_close_formulae() - Price_Option)
        self.asset.St = asset_st
        if plot:
            plt.plot(range_st, list_price, label='Price vs Range', color='blue', linestyle='-', linewidth=2)
            plt.plot(self.asset.St, 0, 'x', label='Option Price at Specific Points',
                     color='red', markersize=10, markeredgewidth=2)
            plt.title('Price of the Option vs. Underlying Asset Price')
            plt.xlabel('Underlying Asset Price (St)')
            plt.ylabel('Option Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        tableau_gains = pd.DataFrame(zip(range_st, list_price), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_gains.index = tableau_gains['Underlying Asset Price (St) move']
        tableau_gains = tableau_gains['gains']
        return tableau_gains
    def Delta_Pnl(self, plot=True):
        range_st = (
            range(round(self.asset.St * 0.9), round(self.asset.St * 1.1), 1)
            if self.asset.St > 10
            else [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 1)]
        )
        asset_st = self.asset.St
        delta = self.Delta_DF()
        delta_pnl = [(delta * (st - asset_st)) for st in range_st]
        self.asset.St = asset_st
        if plot:
            plotPnl(delta_pnl, range_st, 'Delta_PNL')
        tableau_pnl = pd.DataFrame(zip(range_st, delta_pnl), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_pnl.set_index('Underlying Asset Price (St) move', inplace=True)
        return tableau_pnl['gains']
    def Gamma_Pnl(self, plot=True):
        range_st = (
            range(round(self.asset.St * 0.9), round(self.asset.St * 1.1), 1)
            if self.asset.St > 10
            else [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 1)]
        )
        asset_st = self.asset.St
        gamma = self.Gamma_DF()
        gamma_pnl = [(0.5 * gamma * (st - asset_st) ** 2) for st in range_st]
        self.asset.St = asset_st
        if plot:
            plotPnl(gamma_pnl, range_st, 'Gamma_PNL')
        tableau_pnl = pd.DataFrame(zip(range_st, gamma_pnl), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_pnl.set_index('Underlying Asset Price (St) move', inplace=True)
        return tableau_pnl['gains']
    def Third_Order_Pnl(self, plot=True):#Pnl due to Gamma-convexity and Vomma Effect
        range_st = (
            range(round(self.asset.St * 0.9), round(self.asset.St * 1.1), 1)
            if self.asset.St > 10
            else [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 1)]
        )
        asset_st = self.asset.St
        volatility = self.sigma
        speed = self.Speed_DF()
        vega = self.Vega_DF()
        third_order_pnl = [
            (1 / 6 * speed * (st - asset_st) ** 3 + 0.5 * vega * (self.get_sigma() - volatility) ** 2)
            for st in range_st
        ]
        self.asset.St = asset_st
        self.get_sigma()
        if plot:
            plotPnl(third_order_pnl, range_st, '3rd Order PNL')
        tableau_pnl = pd.DataFrame(zip(range_st, third_order_pnl),
                                   columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_pnl.set_index('Underlying Asset Price (St) move', inplace=True)
        return tableau_pnl['gains']
    def nOrderPnl(self, plot=True):
        first_order_pnl = self.Delta_Pnl(plot=False)
        second_order_pnl = self.Gamma_Pnl(plot=False)
        third_order_pnl = self.Third_Order_Pnl(plot=False)
        tableau_pnl = first_order_pnl + second_order_pnl + third_order_pnl
        if plot:
            plotPnl(list(tableau_pnl), tableau_pnl.index, 'N ORDER PNL')
        return tableau_pnl
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
    def run_Booking(self, lot_size:int=None, book_name:str=None): #lot
        if book_name:
            if os.path.basename(os.getcwd()) == 'Quantitative_Finance':
                booking_file_path = f"Booking/Book_Files/{book_name}.xlsx"
            else:
                booking_file_path = f"../Booking/Book_Files/{book_name}.xlsx"
        else:
            if os.path.basename(os.getcwd()) == 'Quantitative_Finance':
                booking_file_path = f"Booking/Book_Files/Booking_history.xlsx"
            else:
                booking_file_path = '../Booking/Book_Files/Booking_history.xlsx'
        booking_file_sheet_name = 'histo_order'
        try:
            df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
        except FileNotFoundError:
            df = pd.DataFrame(columns=['position', 'type', 'quantité', 'maturité', 'asset', 'price asset', 'SP', 'MtM','strike', 'volatility', 'date heure', 'delta', 'gamma', 'vega', 'theta'])
        position = 'long' if self.position>0 else 'short'
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        type = self.type
        quant_save = self.position
        self.position = 1
        try:
            sigma = self.find_vol_impli(self.booked_price)
        except TypeError:
            sigma = self.sigma
        except ValueError:
            sigma = self.sigma
        self.position = quant_save
        self.sigma = self.get_sigma()

        booking = {'position': position, 'type':type, 'quantité':self.position, 'maturité':self.T*365, 'asset':self.asset.name, 'price asset':self.asset.St, 'SP': -self.booked_price * self.position * lot_size if self.booked_price is not None else -self.option_price_close_formulae() * lot_size, 'MtM': self.option_price_close_formulae() * lot_size, 'strike': self.K, 'moneyness %': (self.asset.St / self.K - 1) * 100, 'volatility':sigma, 'date heure':date, 'delta':self.Delta_DF(), 'gamma':self.Gamma_DF(), 'vega':self.Vega_DF(), 'theta':self.Theta_DF()}
        df.loc[len(df)] = booking

        if not os.path.exists(booking_file_path):
            mylogger.logger.warning('Booking file not found')
            mylogger.logger.debug('Booking file creation ...')
            with pd.ExcelWriter(booking_file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=booking_file_sheet_name, index=False)
                mylogger.logger.info(f"Booking file created :{booking_file_path}")
        else:
            with pd.ExcelWriter(booking_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=booking_file_sheet_name, index=False)
                mylogger.logger.info(f"Booking file updated")
        return booking

    def find_vol_impli(self, target_price: float, tol=1e-6, max_iter=100):
        vol = self.sigma
        vol_save = self.sigma
        for i in range(max_iter):
            self.sigma = vol
            price = self.option_price_close_formulae()
            vega = self.Vega_DF()
            """CALIBRATION WITH NEWTON ALGORITHM"""
            vol -= (price - target_price) / (vega * 100)

            if abs(price - target_price) < tol:
                self.sigma = vol_save
                return vol

        raise ValueError("Implied volatility not found within the maximum number of iterations")

    def shift_vol_surface(self, shift: float):
        """
        Shift the entire implied vol surface by 'shift' absolute points.
        If your surface ranges from e.g. 0.2 to 0.5 IV, adding +0.05
        shifts them all to 0.25 -> 0.55.
        """
        if not self.use_vol_surface or self.volatility_surface_df is None:
            return  # no-op or raise an exception if you prefer

        # Shift the columns that contain vol data by 'shift'
        # Here we assume your DF columns with volatility are from column index 1 onwards
        vol_cols = self.volatility_surface_df.columns[1:]
        for col in vol_cols:
            self.volatility_surface_df[col] += shift

        # Re-initialize the interpolator with updated vol_data
        self.set_volatility_surface()


class Option_prem_gen(Option_eu):
    def __init__(self, position, type, asset: (asset_BS), K, T, r, sigma):
        self.position = position
        self.type = type
        self.asset = asset
        self.K = K
        self.t = (len(self.asset.history)-1)/365
        self.T = T
        self.r = r
        self.sigma = sigma
        self.options =[]
        self.positions = []
        if type == "Call Spread":
            self.long_call = Option_eu(1, "Call EU", self.asset, self.K[0], self.t, self.T, self.r, self.sigma)
            self.short_call = Option_eu(-1, "Call EU", self.asset, self.K[1], self.t, self.T, self.r, self.sigma)
            self.positions = [1, -1] #1 : long position, -1 : short position
            self.positions = [i*self.position for i in self.positions]
            self.long_call.position = self.positions[0]
            self.short_call.position = self.positions[1]
            self.options = [self.long_call, self.short_call]

        if type == "Put Spread":
            self.long_put = Option_eu(1, "Put EU", self.asset, self.K[0], self.T, self.r, self.sigma)
            self.short_put = Option_eu(-1, "Put EU", self.asset, self.K[1], self.T, self.r, self.sigma)
            self.positions = [1, -1]
            self.positions = [i*self.position for i in self.positions]
            self.long_put.position = self.positions[0]
            self.short_put.position = self.positions[1]
            self.options = [self.long_put, self.short_put]

        if type == "Strangle":
            self.long_put = Option_eu(1, "Put EU", self.asset, self.K[0], self.T, self.r, self.sigma)
            self.long_call = Option_eu(1, "Call EU", self.asset, self.K[1], self.T, self.r, self.sigma)
            self.positions = [1, 1] #1 : long position, -1 : short position
            self.positions = [i*self.position for i in self.positions]
            self.long_put.position = self.positions[0]
            self.long_call.position = self.positions[1]
            self.options = [self.long_put, self.long_call]


    def update_t(self, days):
        #self.t = days / 365
        for option in self.options:
            option.update_t(days)
        return
    def get_payoff_option(self, ST:int)->int:
        payoff = []
        Purchase = self.option_price_close_formulae()

        for option in self.options:
            if option.type == "Call EU":
                payoff.append(max(ST-option.K, 0) * option.position - Purchase)
            elif option.type == "Put EU":
                payoff.append(max(option.K - ST, 0) * option.position - Purchase)
        payoff_final = sum(payoff)
        return payoff_final
    def display_payoff_option(self, plot=True):
        p
        start = min(self.K)*0.5
        end = max(self.K) * 1.5
        ST = list(range(round(start), round(end)))
        payoffs = []
        for i in ST:
            payoffs.append(self.get_payoff_option(i))
        if plot:
            plot_2d(ST, payoffs, "Asset price", "Payoff", plot=True, title=f"{self.type} payoff")
        return [list(ST), payoffs]

    def option_price_close_formulae(self):
        price_basket_options = 0
        for i in range(len(self.options)):
            price_basket_options+=self.positions[i]*self.options[i].option_price_close_formulae()
        return self.position*price_basket_options

    def Delta_DF(self):
        delta = 0
        for option in self.options:
            delta+= option.Delta_DF()
        return delta
    def Gamma_DF(self):
        gamma = 0
        for option in self.options:
            gamma += option.Gamma_DF()
        return gamma
    def Vega_DF(self):
        vega = 0
        for option in self.options:
            vega += option.Vega_DF()
        return vega
    def Theta_DF(self):
        theta = 0
        for option in self.options:
            theta += option.Theta_DF()
        return theta
    def simu_asset(self, time):
        self.asset.simu_asset(time)
        #self.asset.St = self.asset.history[-1] a reprendre, simuler les sous jacents (list of unique)
        self.t = self.t + time/365
        for option in self.options:
            option.t = self.t

class Option_ame:
    def __init__(self, position, type, asset:(asset_BS), K, T, r, sigma):
        self.position = position
        self.asset = asset
        self.type = type
        self.K = K
        self.t = (len(self.asset.history) - 1) / (365 * 24)
        self.T = T
        self.r = r
        self.sigma = sigma
    def option_price_lsmc(self, Nmc=1000):

        discount_factor = np.exp(-self.r /365)
        t_days = self.t * 365
        T_days = self.T * 365
        asset_paths = np.empty((Nmc, int((T_days-t_days)*24)+1))
        for i in range(Nmc):
            asset_path = simu_actif(self.asset.St, self.t, self.T, self.r, self.sigma)
            asset_paths[i, :] = asset_path
        if self.type == 'Call US':
            payoff = np.maximum(asset_paths[:, -1] - self.K, 0)
        elif self.type == 'Put US':
            payoff = np.maximum(self.K - asset_paths[:, -1], 0)

        option_value = payoff.copy()

        for t in range(len(asset_path) - 1, 0, -1):
            if self.type == 'Call US':
                in_the_money = np.where(asset_paths[:, t] > self.K)[0]
            elif self.type == 'Put US':
                in_the_money = np.where(asset_paths[:, t] < self.K)[0]
            X = asset_paths[in_the_money, t]
            Y = option_value[in_the_money] * discount_factor

            poly = np.polyfit(X, Y, deg=2)
            continuation_value = np.polyval(poly, X)
            if self.type == 'Call US':
                immediate_exercise_value = X - self.K
            elif self.type == 'Put US':
                immediate_exercise_value = self.K - X
            exercise = immediate_exercise_value > continuation_value

            option_value[in_the_money[exercise]] = immediate_exercise_value[exercise]
        option_price = np.mean(option_value) * discount_factor

        return option_price
    def option_price_binomial_tree(self, daily=True):
        steps_per_year = 365 if daily else 1
        volatility = self.sigma / np.sqrt(steps_per_year) if daily else self.sigma

        up = 1 + volatility
        down = 1 - volatility
        discount_factor = np.exp(-self.r / steps_per_year)
        q = (np.exp(self.r / steps_per_year) - down) / (up - down)

        time_to_expiry = int((self.T - self.t) * steps_per_year)

        asset_list = [[self.asset.St]]

        for _ in range(time_to_expiry):
            previous_prices = asset_list[-1]
            new_prices = [price * up for price in previous_prices] + [price * down for price in previous_prices]
            asset_list.append(new_prices)

        if 'Call' in self.type:
            option_values = [max(price - self.K, 0) for price in asset_list[-1]]
        elif 'Put' in self.type:
            option_values = [max(self.K - price, 0) for price in asset_list[-1]]

        for n in range(time_to_expiry - 1, -1, -1):
            option_values = [
                discount_factor * (q * option_values[i] + (1 - q) * option_values[i + 1])
                for i in range(0, len(option_values), 2)
            ]

            if 'Call' in self.type:
                option_values = [max(option_values[i], asset_list[n][i] - self.K) for i in
                                 range(len(option_values))]
            elif 'Put' in self.type:
                option_values = [max(option_values[i], self.K - asset_list[n][i]) for i in
                                 range(len(option_values))]

        return option_values[0]

class ExchangeOption:
    def __init__(self, position, asset1:asset_BS, asset2:asset_BS, T, r, sigma_1, sigma_2, correl):
        self.position = position
        self.asset_1 = asset1
        self.asset_2 = asset2
        self.T = T
        self.r = r
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.correl = correl
    def option_price_close_formulae(self):
        return self.position * close_formulae_magrabe(self.asset_1.St, self.asset_2.St, 0, self.T, self.r, self.sigma_1, self.sigma_2, self.correl)

    def finite_difference_pricing(self, S1_max, S2_max, S1_steps, S2_steps, time_steps):
        """
        Pricing using the finite difference method.
        :param S1_max: Maximum value of asset 1
        :param S2_max: Maximum value of asset 2
        :param S1_steps: Number of grid steps for asset 1
        :param S2_steps: Number of grid steps for asset 2
        :param time_steps: Number of time steps
        :return: Option price
        """
        # Grid initialization
        S1 = np.linspace(0, S1_max, S1_steps)
        S2 = np.linspace(0, S2_max, S2_steps)
        dt = self.T / time_steps
        dS1 = S1[1] - S1[0]
        dS2 = S2[1] - S2[0]

        # Correlation term
        rho = self.correl
        var_1 = self.sigma_1**2
        var_2 = self.sigma_2**2

        # Initialize the option value grid
        V = np.zeros((S1_steps, S2_steps))

        # Boundary condition: payoff at maturity
        for i in range(S1_steps):
            for j in range(S2_steps):
                V[i, j] = max(self.position * (S1[i] - S2[j]), 0)

        # Backward iteration through time
        for t in range(time_steps - 1, -1, -1):
            V_next = V.copy()
            for i in range(1, S1_steps - 1):
                for j in range(1, S2_steps - 1):
                    # Finite difference coefficients
                    dVdS1 = (V_next[i + 1, j] - V_next[i - 1, j]) / (2 * dS1)
                    dVdS2 = (V_next[i, j + 1] - V_next[i, j - 1]) / (2 * dS2)
                    d2VdS1 = (V_next[i + 1, j] - 2 * V_next[i, j] + V_next[i - 1, j]) / (dS1**2)
                    d2VdS2 = (V_next[i, j + 1] - 2 * V_next[i, j] + V_next[i, j - 1]) / (dS2**2)
                    d2VdS1S2 = (V_next[i + 1, j + 1] - V_next[i + 1, j - 1] -
                                V_next[i - 1, j + 1] + V_next[i - 1, j - 1]) / (4 * dS1 * dS2)

                    # PDE discretization
                    V[i, j] = V_next[i, j] + dt * (
                        -self.r * V_next[i, j]
                        + (self.r - 0.5 * var_1) * S1[i] * dVdS1
                        + (self.r - 0.5 * var_2) * S2[j] * dVdS2
                        + 0.5 * var_1 * S1[i]**2 * d2VdS1
                        + 0.5 * var_2 * S2[j]**2 * d2VdS2
                        + rho * self.sigma_1 * self.sigma_2 * S1[i] * S2[j] * d2VdS1S2
                    )

        # Return interpolated price for initial asset prices
        S1_idx = np.searchsorted(S1, self.asset_1.St)
        S2_idx = np.searchsorted(S2, self.asset_2.St)
        return V[S1_idx, S2_idx]
if __name__ == '__main__':
    # Create assets and the exchange option
    stock1 = asset_BS(100, 0)  # S1 = 100
    stock2 = asset_BS(95, 0)  # S2 = 101
    spread_option = ExchangeOption(1, stock1, stock2, 365 / 365, 0.05, 0.2, 0.25, 0.5)

    # Price the option using FDM
    price_fd = spread_option.finite_difference_pricing(S1_max=150, S2_max=150, S1_steps=200, S2_steps=200, time_steps=500)
    print("FDM Spread Option Price:", price_fd)
    print("FDM Spread Option Price:", spread_option.option_price_close_formulae())

    # price1 = optionUS.option_price_lsmc(Nmc=1000)
    # optionEU = Option_eu(1, 'Call EU', stock1, 100, 10/365, 0.05, 0.2)
    # price1 = optionEU.option_price_binomial_tree()
    # optionUS = Option_ame(1, 'Put US', stock1, 100, 25/365, 0.05, 0.5)
    # optionUS.option_price_binomial_tree()
    # optionUS.option_price_lsmc()
