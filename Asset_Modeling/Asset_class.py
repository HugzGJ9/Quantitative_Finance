from Asset_Modeling.Actif_stoch_BS import simu_actif
from Graphics.Graphics import plot_2d
import pandas as pd
import datetime
class asset_BS():
    def __init__(self, S0, quantity, name=None, mu=None, sigma = None):
        self.S0 = S0
        self.St = S0
        self.quantity = quantity
        self.name = name
        self.history = [S0]
        self.mu = mu if mu is not None else 0.1
        self.sigma = sigma if sigma is not None else 0.1
        self.t = 0
    def simu_asset(self, T)->None:
        St = simu_actif(self.St, self.t, T, self.mu, self.sigma)
        St.pop(0)
        for st in St:
            self.history.append(st)
        self.St = self.history[-1]
        self.t = self.t + T / 365
    def plot(self):
        plot_2d(list(range(len(self.history))), self.history, title='Asset Price Path', x_axis='t', y_axis='asset price', isShow=True)
    def Delta_DF(self):
        return 1*self.quantity
    def Gamma_DF(self):
        return 0
    def Vega_DF(self):
        return 0
    def Theta_DF(self):
        return 0
    def run_Booking(self, lot_size):
        booking_file_path = 'Booking_history.xlsx'
        booking_file_sheet_name = 'histo_order'

        df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
        position = 'long' if self.quantity>0 else 'short'
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        type = 'asset'
        booking = {'position': position, 'type':type, 'quantité':self.quantity, 'maturité':None, 'asset':self.name, 'price asset':self.St, 's-p': -self.St*lot_size*self.quantity, 'MtM': self.St*lot_size*self.quantity,'strike': None, 'vol':None, 'vol ST':None, 'date heure':date, 'delta':self.Delta_DF(), 'gamma':None, 'vega':None, 'theta':None}
        df.loc[len(df)] = booking
        with pd.ExcelWriter(booking_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=booking_file_sheet_name, index=False)
        return

if __name__ == '__main__':
    stock1 = asset_BS(2.240, 0, mu=0.1, sigma=0.2)
    stock1.simu_asset(1/365)
    stock1.plot()