from BM_def import BM
import numpy as np
from Graphics import plot_2d
import matplotlib.pyplot as plt
from Actif_stoch_BS import simu_actif
from Actif_stoch_BS import option_eu_mc
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from ttkthemes import ThemedTk
from payoffs import payoff_call_eu, payoff_put_eu

class Option_eu:
    def __init__(self, type, St, K, T, sigma, root=None):
        self.type = type
        self.St = St
        self.K = K
        self.T = T
        self.sigma = sigma

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
        self.option_type_combobox = ttk.Combobox(self.frame, textvariable=self.option_type_var, values=["Call", "Put"])
        self.option_type_combobox.grid(row=1, column=1, pady=5)

        self.St_label = ttk.Label(self.frame, text="Current Stock Price:")
        self.St_label.grid(row=2, column=0, sticky="w")

        self.St_entry = ttk.Entry(self.frame)
        self.St_entry.grid(row=2, column=1, pady=5)

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
                                           command=self.option_eu_mc)
        self.calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(self.frame, text="", font=("Helvetica", 14))
        self.result_label.grid(row=7, column=0, columnspan=2, pady=10)


    def option_eu_mc(self):
        prix_option = 0
        Nmc = 10000
        for i in range(Nmc):
            prix_actif = simu_actif(self.St, 100, self.T, 0.1, self.sigma)
            if self.type == "Call":
                prix_option += payoff_call_eu(prix_actif[-1], K)
            elif self.type == "Put":
                prix_option += payoff_put_eu(prix_actif[-1], K)
        prix_option = prix_option / Nmc

        self.result_label.config(text=f"Option Price: {prix_option:.4f}")
        return prix_option




if __name__ == '__main__':
    N = 100
    T = 1
    t = 0
    K = 105
    St = 100
    Nmc = 1000000
    time = np.linspace(0, T, N+1)
    title = f'Simulation de {Nmc} prix d actif'
    x_axis = 'Temps'
    y_axis = 'Valeur'
    root = ThemedTk(theme="breeze")
    #to activate the user interface
    # root.mainloop()
    call_eu1 = Option_eu('Call', St, K, T, 0.3, root)
    print(call_eu1.option_eu_mc())
    # print(option_eu_mc(St, T, K, Nmc))

    # for i in range(Nmc):
    #     St = simu_actif(10, N, T, 0.1, 0.3)
    #     plot_2d(time, St, title, x_axis, y_axis, False)
    # plt.show()

      # Choose your preferred theme



