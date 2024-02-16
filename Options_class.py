import numpy as np
from matplotlib import pyplot as plt

from Asset_class import asset_BS
from Actif_stoch_BS import simu_actif
import tkinter as tk
from tkinter import ttk

from Graphics import display_payoff_eu, plot_2d
from payoffs import payoff_call_eu, payoff_put_eu, payoff_call_asian, payoff_put_asian, close_formulae_call_eu, close_formulae_put_eu, delta_option_eu, gamma_option_eu
class Option_eu:
    #root parameter to
    def __init__(self, position, type, asset:(asset_BS), K, t, T, r, sigma, root=None):
        self.position = position
        self.asset = asset
        self.type = type
        self.K = K
        self.t = t
        self.T = T
        self.r = r
        self.sigma = sigma

        if root!= None:
            self.root = root
            self.root.title("European Option Pricing")
            self.root.geometry("400x400")

            self.style = ttk.Style(self.root)
            self.style.theme_use("plastik")

            self.frame = ttk.Frame(root)
            self.frame.pack(pady=20, padx=20)

            self.label = ttk.Label(self.frame, text="European Option Pricing", font=("Helvetica", 16))
            self.label.grid(row=0, column=0, columnspan=2, pady=10)

            self.option_type_label = ttk.Label(self.frame, text="Option Type:")
            self.option_type_label.grid(row=1, column=0, sticky="w")

            self.option_type_var = tk.StringVar()
            self.option_type_combobox = ttk.Combobox(self.frame, textvariable=self.option_type_var, values=["Call EU", "Put EU", "Call Asian", "Put Asian"])
            self.option_type_combobox.grid(row=1, column=1, pady=5)

            self.asset.St_label = ttk.Label(self.frame, text="Current Stock Price:")
            self.asset.St_label.grid(row=2, column=0, sticky="w")

            self.asset.St_entry = ttk.Entry(self.frame)
            self.asset.St_entry.grid(row=2, column=1, pady=5)

            self.K_label = ttk.Label(self.frame, text="Strike Price:")
            self.K_label.grid(row=3, column=0, sticky="w")

            self.K_entry = ttk.Entry(self.frame)
            self.K_entry.grid(row=3, column=1, pady=5)

            self.T_label = ttk.Label(self.frame, text="Time to Maturity (T):")
            self.T_label.grid(row=4, column=0, sticky="w")

            self.T_entry = ttk.Entry(self.frame)
            self.T_entry.grid(row=4, column=1, pady=5)

            self.sigma_label = ttk.Label(self.frame, text="Volatility (sigma):")
            self.sigma_label.grid(row=5, column=0, sticky="w")

            self.sigma_entry = ttk.Entry(self.frame)
            self.sigma_entry.grid(row=5, column=1, pady=5)

            self.calculate_button = ttk.Button(self.frame, text="Calculate Option Price",
                                               command=self.option_price_close_formulae)
            self.calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

            self.result_label = ttk.Label(self.frame, text="", font=("Helvetica", 14))
            self.result_label.grid(row=7, column=0, columnspan=2, pady=10)
    def get_payoff_option(self, ST:int)->int:

        if self.type == "Call EU":
            payoff = payoff_call_eu(ST, self.K) * self.position
        elif self.type == "Put EU":
            payoff = payoff_put_eu(ST, self.K) * self.position
        return payoff
    def display_payoff_option(self):
        start = self.K*0.5
        end = self.K*1.5
        ST = list(range(round(start), round(end)))
        payoffs = []
        for i in ST:
            payoffs.append(self.get_payoff_option(i))
        plot_2d(ST, payoffs, f"{self.type} payoff", "Asset price", "Payoff", isShow=True)
    def option_price_close_formulae(self):
        if self.type == "Call EU":
            option_price = close_formulae_call_eu(self.asset.St, self.K, self.t, self.T, self.r, self.sigma)
            return self.position*option_price
        elif self.type == "Put EU":
            option_price = close_formulae_put_eu(self.asset.St, self.K, self.t, self.T, self.r, self.sigma)
            return self.position*option_price
        # elif self.type == "Call Spread":
        #     long_call = close_formulae_call_eu(self.asset.St, self.K, self.t, self.T, self.r, self.sigma)
        #     short_call = close_formulae_call_eu(self.asset.St, self.K_2, self.t, self.T, self.r, self.sigma)
        #     option_price = long_call - short_call
        #     return option_price
    def option_price_mc(self):
        prix_option = 0
        Nmc = 5000
        for i in range(Nmc):
            t_days = self.t*365
            T_days = self.T*365
            prix_actif = simu_actif(self.asset.St, t_days, T_days, self.r, self.sigma)
            if self.type == "Call EU":
                prix_option += payoff_call_eu(prix_actif[-1], self.K)
            elif self.type == "Put EU":
                prix_option += payoff_put_eu(prix_actif[-1], self.K)
            elif self.type == "Call Asian":
                prix_option += payoff_call_asian(prix_actif, self.K)
            elif self.type == "Put Asian":
                prix_option += payoff_put_asian(prix_actif, self.K)
        prix_option = np.exp(-self.r*(self.T-self.t))*prix_option / Nmc

        # self.result_label.config(text=f"Option Price: {prix_option:.4f}")
        return self.position*prix_option

    # def Delta(self):
    #     option_delta = (delta_option_eu(self.position, self.type, self.asset, self.K, self.t, self.T, self.r, self.sigma))
    #     return option_delta
    def Delta_DF(self):
        delta_St = 0.00001
        asset_delta = asset_BS(self.asset.St + delta_St, self.asset.quantity)
        option_delta_St = Option_eu(self.position, self.type, asset_delta, self.K, self.t, self.T, self.r, self.sigma).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type,self.asset, self.K, self.t, self.T, self.r, self.sigma).option_price_close_formulae()

        delta = (option_delta_St - option_option)/delta_St
        return delta
    # def Gamma(self):
    #     option_gamma = (gamma_option_eu(self.position, self.type, self.asset, self.K, self.t, self.T, self.r, self.sigma))
    #     return option_gamma
    def Gamma_DF(self):
        delta_St = 0.00001
        asset_delta = asset_BS(self.asset.St + delta_St, self.asset.quantity)
        asset_delta_neg = asset_BS(self.asset.St - delta_St, self.asset.quantity)
        option_gamma_plus = Option_eu(self.position, self.type, asset_delta, self.K, self.t, self.T, self.r,
                                      self.sigma).option_price_close_formulae()
        option_gamma_minus = Option_eu(self.position, self.type, asset_delta_neg, self.K, self.t, self.T, self.r,
                                       self.sigma).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type,self.asset, self.K, self.t, self.T, self.r, self.sigma).option_price_close_formulae()

        gamma = ((option_gamma_plus + option_gamma_minus - 2 * option_option) / delta_St ** 2)
        return gamma
    def Vega_DF(self):
        delta_vol = 0.00001
        option_delta_vol = Option_eu(self.position, self.type, self.asset, self.K, self.t, self.T, self.r,
                                    self.sigma+delta_vol).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type, self.asset, self.K, self.t, self.T, self.r,
                                  self.sigma).option_price_close_formulae()

        vega = (option_delta_vol - option_option) / delta_vol
        return vega/100
    def Theta_DF(self):
        delta_t = 0.00001
        option_delta_t = Option_eu(self.position, self.type, self.asset, self.K, self.t+delta_t, self.T, self.r,
                                    self.sigma).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type, self.asset, self.K, self.t, self.T, self.r,
                                  self.sigma).option_price_close_formulae()

        theta = (option_delta_t - option_option) / delta_t
        return theta/365.6

    def simu_asset(self, time):
        self.asset.simu_asset(time)
        #self.asset.St = self.asset.history[-1]
        self.t = self.t + time/365.6


class Option_prem_gen(Option_eu):
    def __init__(self, position, type, asset:(asset_BS), K, t, T, r, sigma, root=None):
        self.position = position
        self.type = type
        self.asset = asset
        self.asset.St = asset.St
        self.K = K
        self.t = t
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
            self.long_put = Option_eu(1, "Put EU", self.asset, self.K[0], self.t, self.T, self.r, self.sigma)
            self.short_put = Option_eu(-1, "Put EU", self.asset, self.K[1], self.t, self.T, self.r, self.sigma)
            self.positions = [1, -1]
            self.positions = [i*self.position for i in self.positions]
            self.long_put.position = self.positions[0]
            self.short_put.position = self.positions[1]
            self.options = [self.long_put, self.short_put]

        if type == "Strangle":
            self.long_put = Option_eu(1, "Put EU", self.asset, self.K[0], self.t, self.T, self.r, self.sigma)
            self.long_call = Option_eu(1, "Call EU", self.asset, self.K[1], self.t, self.T, self.r, self.sigma)
            self.positions = [1, 1] #1 : long position, -1 : short position
            self.positions = [i*self.position for i in self.positions]
            self.long_put.position = self.positions[0]
            self.long_call.position = self.positions[1]
            self.options = [self.long_put, self.long_call]


    '''def option_price_close_formulae(self):
        if self.type == "Call Spread":
            Call1_price = self.long_call.option_price_close_formulae()
            Call2_price = self.short_call.option_price_close_formulae()
            return (Call1_price - Call2_price)'''
    def get_payoff_option(self, ST:int)->int:
        payoff = []
        for option in self.options:
            if option.type == "Call EU":
                payoff.append(max(ST-option.K, 0) * option.position)
            elif option.type == "Put EU":
                payoff.append(max(option.K - ST, 0) * option.position)
        payoff_final = sum(payoff)
        return payoff_final
    def display_payoff_option(self):
        start = min(self.K)*0.5
        end = max(self.K) * 1.5
        ST = list(range(round(start), round(end)))
        payoffs = []
        for i in ST:
            payoffs.append(self.get_payoff_option(i))
        plot_2d(ST, payoffs, f"{self.type} payoff", "Asset price", "Payoff", isShow=True)

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

def plot_greek_curves_2d(position, type_option, greek, K, t_, T, r, vol):
    St_range = range(20, 180, 1)
    Eu_options = ['Call EU', 'Put EU']
    Option_first_gen = ['Call Spread', 'Put Spread', 'Strangle']
    if type_option in Eu_options:
        Option = Option_eu
    elif type_option in Option_first_gen:
        Option = Option_prem_gen
    if greek.lower() =='delta':
        Option.greek = Option.Delta_DF
    elif greek.lower() == 'gamma':
        Option.greek = Option.Gamma_DF
    elif greek.lower() == 'vega':
        Option.greek = Option.Vega_DF
    elif greek.lower() == 'theta':
        Option.greek = Option.Theta_DF

    if type(vol) == list:
        moving_param = vol
        moving_param_label = "volatility"
    elif type(T) == list:
        moving_param = T
        moving_param_label = "maturity"
    elif type(r) == list:
        moving_param = r
        moving_param_label = "st rate"
    else:
        greek_list = []
        for i in St_range:
            asset = asset_BS(i, 0)
            option_obj = Option(position, type_option, asset, K, t_, T, r, vol)
            greek_list.append(option_obj.greek())
        #greek_list = [i*position for i in greek_list]
        plot_2d(St_range, greek_list, f"{greek} curve", "Prix de l'actif", greek, True)
        return

    for v in moving_param:
        if moving_param_label == "volatility":
            vol = v
        elif moving_param_label == "maturity":
            T = v
        elif moving_param_label == "st rate":
            r = v
        greek_list = []
        for i in St_range:
            asset = asset_BS(i, 0)
            option_obj = Option(position, type_option, asset, K, t_, T, r, vol)
            greek_list.append(option_obj.greek())
        greek_list = [i*position for i in greek_list]
        plot_2d(St_range, greek_list, f"{greek} curve - {type_option}", "Prix de l'actif", greek, False)
    moving_param = [moving_param_label + ' : ' + str(x) for x in moving_param]
    plt.legend(moving_param)
    plt.show()