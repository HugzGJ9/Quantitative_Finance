import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Asset_Modeling.Asset_class import asset_BS
from  Asset_Modeling.Actif_stoch_BS import simu_actif
import tkinter as tk
from tkinter import ttk

from Graphics.Graphics import plot_2d
from Options.payoffs import payoff_call_eu, payoff_put_eu, payoff_call_asian, payoff_put_asian, close_formulae_call_eu, \
    close_formulae_put_eu, payoff_call_eu_barrier

class Option_eu:
    #root parameter to
    def __init__(self, position, type, asset:(asset_BS), K, T, r, sigma, barrier=None, root=None):
        self.position = position
        self.asset = asset
        self.type = type
        self.K = K
        self.t = (len(self.asset.history)-1)/(365.6*24)
        self.T = T
        self.r = r
        self.sigma = sigma
        self.barrier = barrier
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

    def update_t(self, days):
        self.t = days/365.6
        return
    def get_payoff_option(self, ST:int)->int:

        if self.type == "Call EU":
            payoff = payoff_call_eu(ST, self.K) * self.position
        elif self.type == "Put EU":
            payoff = payoff_put_eu(ST, self.K) * self.position
        return payoff
    def display_payoff_option(self):

        if self.K>10:
            ST = range(round(self.K*0.5), round(self.K*1.5), 2)
        else:
            ST = [x/100 for x in range(round(self.K*0.5*100), round(self.K*3*100), 2)]

        payoffs = []
        for i in ST:
            payoffs.append(self.get_payoff_option(i))
        plot_2d(ST, payoffs, "Asset price", "Payoff", isShow=True, title=f"{self.type} payoff")
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
            t_days = self.t*365.6
            T_days = self.T*365.6
            prix_actif = simu_actif(self.asset.St, t_days, T_days, self.r, self.sigma)
            if self.type == "Call EU":
                prix_option += payoff_call_eu(prix_actif[-1], self.K)
            elif self.type == "Put EU":
                prix_option += payoff_put_eu(prix_actif[-1], self.K)
            elif self.type == "Call Asian":
                prix_option += payoff_call_asian(prix_actif, self.K)
            elif self.type == "Put Asian":
                prix_option += payoff_put_asian(prix_actif, self.K)
            elif self.type == "Call In & Out":
                activation_barrier = True in [self.barrier<St for St in prix_actif]
                prix_option += payoff_call_eu_barrier(prix_actif[-1], self.K, activation_barrier)
        prix_option = np.exp(-self.r*(self.T-self.t))*prix_option / Nmc

        # self.result_label.config(text=f"Option Price: {prix_option:.4f}")
        return self.position*prix_option

    # def Delta(self):
    #     option_delta = (delta_option_eu(self.position, self.type, self.asset, self.K, self.t, self.T, self.r, self.sigma))
    #     return option_delta
    def Delta_DF(self):
        delta_St = 0.00001
        asset_delta = asset_BS(self.asset.St + delta_St, self.asset.quantity)
        option_delta_St = Option_eu(self.position, self.type, asset_delta, self.K, self.T, self.r, self.sigma).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type,self.asset, self.K, self.T, self.r, self.sigma).option_price_close_formulae()

        delta = (option_delta_St - option_option)/delta_St
        return delta
    def DeltaRisk(self):
        Delta_Option = self.Delta_DF()
        if self.asset.St>10:
            range_st = range(round(self.asset.St*0.5), round(self.asset.St*1.5), 2)
        else:
            range_st = [x/100 for x in range(round(self.asset.St*0.5*100), round(self.asset.St*3*100), 2)]
        asset_st = self.asset.St
        list_delta = list()
        for st in range_st:
            self.asset.St = st
            list_delta.append(self.Delta_DF())
        self.asset.St = asset_st
        plt.plot(range_st, list_delta, label='Delta vs Range', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.asset.St, Delta_Option, 'x', label='Option Delta at Specific Points',
                 color='red', markersize=10, markeredgewidth=2)

        # Adding titles and labels
        plt.title('Delta of the Option vs. Underlying Asset Price')
        plt.xlabel('Underlying Asset Price (St)')
        plt.ylabel('Option Delta')

        # Adding a legend
        plt.legend()

        # Adding grid for better readability
        plt.grid(True)

        # Displaying the plot
        plt.show()
        tableau_delta = pd.DataFrame(zip(range_st, list_delta), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_delta.index = tableau_delta['Underlying Asset Price (St) move']
        tableau_delta = tableau_delta['gains']
        return tableau_delta
    def Delta_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = self.T
        list_delta = []

        range_t = [t / (365.6*100) for t in range(0, round(self.T * 100*365.6), 2)]

        for t in range_t:
            self.T = option_matu - t
            for st in range_st:
                self.asset.St = st
                list_delta.append(self.Delta_DF())

        self.asset.St = asset_st
        self.T = option_matu

        range_st_mesh, range_t_mesh = np.meshgrid(range_st, range_t)
        list_delta = np.array(list_delta).reshape(len(range_t), len(range_st))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(range_st_mesh, range_t_mesh*365.6, list_delta, cmap='magma')
        ax.set_title('Delta of the Option vs. Underlying Asset Price and Time')
        ax.set_xlabel('Underlying Asset Price (St)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Option Delta')
        plt.show()

    def Gamma_DF(self):
        delta_St = 0.00001
        asset_delta = asset_BS(self.asset.St + delta_St, self.asset.quantity)
        asset_delta_neg = asset_BS(self.asset.St - delta_St, self.asset.quantity)
        option_gamma_plus = Option_eu(self.position, self.type, asset_delta, self.K, self.T, self.r,
                                      self.sigma).option_price_close_formulae()
        option_gamma_minus = Option_eu(self.position, self.type, asset_delta_neg, self.K, self.T, self.r,
                                       self.sigma).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type,self.asset, self.K, self.T, self.r, self.sigma).option_price_close_formulae()

        gamma = ((option_gamma_plus + option_gamma_minus - 2 * option_option) / delta_St ** 2)
        return gamma

    def GammaRisk(self):
        Gamma_Option = self.Gamma_DF()

        if self.asset.St>10:
            range_st = range(round(self.asset.St*0.5), round(self.asset.St*1.5), 2)
        else:
            range_st = [x/100 for x in range(round(self.asset.St*0.5*100), round(self.asset.St*3*100), 2)]
        asset_st = self.asset.St
        list_gamma = list()
        for st in range_st:
            self.asset.St = st
            list_gamma.append(self.Gamma_DF())
        self.asset.St = asset_st
        plt.plot(range_st, list_gamma, label='Gamma vs Range', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.asset.St, Gamma_Option, 'x', label='Option Gama at Specific Points',
                 color='red', markersize=10, markeredgewidth=2)

        # Adding titles and labels
        plt.title('Gamma of the Option vs. Underlying Asset Price')
        plt.xlabel('Underlying Asset Price (St)')
        plt.ylabel('Option Gamma')

        # Adding a legend
        plt.legend()

        # Adding grid for better readability
        plt.grid(True)

        # Displaying the plot
        plt.show()
        return

    def Gamma_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = self.T
        list_gamma = []

        range_t = [t / (365.6*100) for t in range(0, round(self.T * 100*365.6), 2)]

        for t in range_t:
            self.T = option_matu - t
            for st in range_st:
                self.asset.St = st
                list_gamma.append(self.Gamma_DF())

        self.asset.St = asset_st
        self.T = option_matu

        range_st_mesh, range_t_mesh = np.meshgrid(range_st, range_t)
        list_gamma = np.array(list_gamma).reshape(len(range_t), len(range_st))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(range_st_mesh, range_t_mesh*365.6, list_gamma, cmap='magma')
        ax.set_title('Gamma of the Option vs. Underlying Asset Price and Time')
        ax.set_xlabel('Underlying Asset Price (St)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Option Gamma')
        plt.show()
    def Vega_DF(self):
        delta_vol = 0.00001
        option_delta_vol = Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r,
                                    self.sigma+delta_vol).option_price_close_formulae()
        option_option = Option_eu(self.position, self.type, self.asset, self.K, self.T, self.r,
                                  self.sigma).option_price_close_formulae()

        vega = (option_delta_vol - option_option) / delta_vol
        return vega/100

    def VegaRisk(self):
        Vega_Option = self.Vega_DF()
        if self.asset.St>10:
            range_st = range(round(self.asset.St*0.5), round(self.asset.St*1.5), 2)
        else:
            range_st = [x/100 for x in range(round(self.asset.St*0.5*100), round(self.asset.St*3*100), 2)]
        asset_st = self.asset.St
        list_vega = list()
        for st in range_st:
            self.asset.St = st
            list_vega.append(self.Vega_DF())
        self.asset.St = asset_st
        plt.plot(range_st, list_vega, label='Vega vs Range', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.asset.St, Vega_Option, 'x', label='Option Vega at Specific Points',
                 color='red', markersize=10, markeredgewidth=2)

        # Adding titles and labels
        plt.title('Vega of the Option vs. Underlying Asset Price')
        plt.xlabel('Underlying Asset Price (St)')
        plt.ylabel('Option Vega')

        # Adding a legend
        plt.legend()

        # Adding grid for better readability
        plt.grid(True)

        # Displaying the plot
        plt.show()
        return
    def Vega_surface(self):


        vol_option = self.sigma
        option_matu = self.T
        list_vega = []
        range_sigma = [sigma / 100 for sigma in range(round(self.sigma * 100*0.5), round(self.sigma * 100*1.5), 2)]
        range_t = [t / (365.6*100) for t in range(0, round(self.T * 100*365.6), 2)]

        for t in range_t:
            self.T = option_matu - t
            for sigma_ in range_sigma:
                self.sigma = sigma_
                list_vega.append(self.Vega_DF())

        self.sigma = vol_option
        self.T = option_matu

        range_sigma_mesh, range_t_mesh = np.meshgrid(range_sigma, range_t)
        list_vega = np.array(list_vega).reshape(len(range_t), len(range_sigma))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(range_sigma_mesh, range_t_mesh*365.6, list_vega, cmap='magma')
        ax.set_title('Vega of the Option vs. Underlying Asset Price and Time')
        ax.set_xlabel('Volatility (sigma)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Option Vega')
        plt.show()
    def Theta_DF(self):
        delta_t = 0.00001
        self.t = self.t+delta_t
        option_delta_t = self.option_price_close_formulae()
        self.t = self.t-delta_t

        option_option = self.option_price_close_formulae()

        theta = (option_delta_t - option_option) / delta_t
        return theta/365.6
    def ThetaRisk(self):
        Theta_Option = self.Theta_DF()
        if self.asset.St>10:
            range_st = range(round(self.asset.St*0.5), round(self.asset.St*1.5), 2)
        else:
            range_st = [x/100 for x in range(round(self.asset.St*0.1*100), round(self.asset.St*3*100), 2)]
        asset_st = self.asset.St
        list_theta = list()
        for st in range_st:
            self.asset.St = st
            list_theta.append(self.Theta_DF())
        self.asset.St = asset_st
        plt.plot(range_st, list_theta, label='Theta vs Range', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.asset.St, Theta_Option, 'x', label='Option Theta at Specific Points',
                 color='red', markersize=10, markeredgewidth=2)

        # Adding titles and labels
        plt.title('Theta of the Option vs. Underlying Asset Price')
        plt.xlabel('Underlying Asset Price (St)')
        plt.ylabel('Option Theta')

        # Adding a legend
        plt.legend()

        # Adding grid for better readability
        plt.grid(True)

        # Displaying the plot
        plt.show()
        return

    def Theta_surface(self):
        if self.asset.St > 10:
            range_st = np.arange(round(self.asset.St * 0.5), round(self.asset.St * 1.5), 2)
        else:
            range_st = [x / 100 for x in range(round(self.asset.St * 0.8 * 100), round(self.asset.St * 1.2 * 100), 2)]

        asset_st = self.asset.St
        option_matu = self.T
        list_theta = []

        range_t = [t / (365.6*100) for t in range(0, round(self.T * 100*365.6), 2)]

        for t in range_t:
            self.T = option_matu - t
            for st in range_st:
                self.asset.St = st
                list_theta.append(self.Theta_DF())

        self.asset.St = asset_st
        self.T = option_matu

        range_st_mesh, range_t_mesh = np.meshgrid(range_st, range_t)
        list_theta = np.array(list_theta).reshape(len(range_t), len(range_st))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(range_st_mesh, range_t_mesh*365.6, list_theta, cmap='magma')
        ax.set_title('Theta of the Option vs. Underlying Asset Price and Time')
        ax.set_xlabel('Underlying Asset Price (St)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Option Theta')
        plt.show()
    def simu_asset(self, time):
        self.asset.simu_asset(time)
        #self.asset.St = self.asset.history[-1]
        self.t = self.t + time/365.6

    def PnlRisk(self):
        Price_Option = self.option_price_close_formulae()
        # range_st = np.arange(self.asset.St-3, self.asset.St+3, 0.1)

        if self.asset.St>10:
            range_st = range(round(self.asset.St*0.5), round(self.asset.St*1.5), 2)
        else:
            range_st = [x/100 for x in range(round((self.asset.St - 1 ) *100), round((self.asset.St + 1 ) *100), 2)]

        asset_st = self.asset.St
        list_price = list()
        for st in range_st:
            self.asset.St = st
            list_price.append(self.option_price_close_formulae() - Price_Option)
        self.asset.St = asset_st
        plt.plot(range_st, list_price, label='Price vs Range', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.asset.St, 0, 'x', label='Option Price at Specific Points',
                 color='red', markersize=10, markeredgewidth=2)

        # Adding titles and labels
        plt.title('Price of the Option vs. Underlying Asset Price')
        plt.xlabel('Underlying Asset Price (St)')
        plt.ylabel('Option Price')

        # Adding a legend
        plt.legend()

        # Adding grid for better readability
        plt.grid(True)

        # Displaying the plot
        plt.show()
        tableau_gains = pd.DataFrame(zip(range_st, list_price), columns=['Underlying Asset Price (St) move', 'gains'])
        tableau_gains.index = tableau_gains['Underlying Asset Price (St) move']
        tableau_gains = tableau_gains['gains']
        return tableau_gains
    def RiskAnalysis(self):
        self.DeltaRisk()
        self.GammaRisk()
        self.VegaRisk()
        self.ThetaRisk()
        return
    def run_Booking(self, lot_size:int=None): #lot
        booking_file_path = 'Booking_history.xlsx'
        booking_file_sheet_name = 'histo_order'

        df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
        position = 'long' if self.position>0 else 'short'
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        type = self.type
        booking = {'position': position, 'type':type, 'quantité':self.position, 'maturité':self.T*365.6, 'asset':self.asset.name, 'price asset':self.asset.St, 's-p': -self.option_price_close_formulae() * lot_size, 'MtM': self.option_price_close_formulae() * lot_size, 'strike': self.K, 'moneyness %': (self.asset.St / self.K - 1) * 100, 'vol':self.sigma, 'vol ST':None, 'date heure':date, 'delta':self.Delta_DF(), 'gamma':self.Gamma_DF(), 'vega':self.Vega_DF(), 'theta':self.Theta_DF()}
        df.loc[len(df)] = booking
        with pd.ExcelWriter(booking_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=booking_file_sheet_name, index=False)
        return

class Option_prem_gen(Option_eu):
    def __init__(self, position, type, asset:(asset_BS), K, T, r, sigma, root=None):
        self.position = position
        self.type = type
        self.asset = asset
        self.K = K
        self.t = (len(self.asset.history)-1)/365.6
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
    def update_t(self, days):
        #self.t = days / 365.6
        for option in self.options:
            option.update_t(days)
        return
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
        plot_2d(ST, payoffs, "Asset price", "Payoff", isShow=True, title=f"{self.type} payoff")

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
        plot_2d(St_range, greek_list, "Prix de l'actif", greek, True, f"{greek} curve")
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
        plot_2d(St_range, greek_list, "Prix de l'actif", greek, False, f"{greek} curve - {type_option}")
    moving_param = [moving_param_label + ' : ' + str(x) for x in moving_param]
    plt.legend(moving_param)
    plt.show()