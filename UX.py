import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from ttkthemes import ThemedTk
from Actif_stoch_BS import simu_actif

class OptionEUApp:
    def __init__(self, root):
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

        self.calculate_button = ttk.Button(self.frame, text="Calculate Option Price", command=self.calculate_option_price)
        self.calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(self.frame, text="", font=("Helvetica", 14))
        self.result_label.grid(row=7, column=0, columnspan=2, pady=10)

    def calculate_option_price(self):
        option_type = self.option_type_var.get()
        St = float(self.St_entry.get())
        K = float(self.K_entry.get())
        T = float(self.T_entry.get())
        sigma = float(self.sigma_entry.get())

        Nmc = 10000
        prix_option = 0

        for i in range(Nmc):
            prix_actif = simu_actif(St, 100, T, 0.1, sigma)
            if option_type == "Call":
                prix_option += max(prix_actif[-1] - K, 0)
            else:
                prix_option += max(K - prix_actif[-1], 0)

        prix_option = prix_option / Nmc
        self.result_label.config(text=f"Option Price: {prix_option:.4f}")


if __name__ == '__main__':
    root = ThemedTk(theme="breeze")  # Choose your preferred theme
    app = OptionEUApp(root)
    root.mainloop()
