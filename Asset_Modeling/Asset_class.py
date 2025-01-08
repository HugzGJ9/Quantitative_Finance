import os

from Asset_Modeling.Actif_stoch_BS import simu_actif
from Graphics.Graphics import plot_2d
import pandas as pd
import datetime
from Logger.Logger import mylogger
class asset_BS():
    def __init__(self, S0, quantity, name=None, mu=None, sigma = None, logger=False):
        self.S0 = S0
        self.St = S0
        self.quantity = quantity
        self.name = name
        self.history = [S0]
        self.mu = mu if mu is not None else 0.1
        self.sigma = sigma if sigma is not None else 0.1
        self.t = 0
        if logger:
            mylogger.logger.info(f"Asset has been intiated")

    def simu_asset(self, T)->None:
        St = simu_actif(self.St, self.t, T, self.mu, self.sigma)
        St.pop(0)
        for st in St:
            self.history.append(st)
        self.St = self.history[-1]
        self.t = self.t + T / 365
    def plot(self):
        plot_2d(list(range(len(self.history))), self.history, title='Asset Price Path', x_axis='t', y_axis='asset price', plot=True)
    def Delta_DF(self):
        return 1*self.quantity
    def Gamma_DF(self):
        return 0
    def Vega_DF(self):
        return 0
    def Theta_DF(self):
        return 0
    def run_Booking(self, lot_size, book_name:str=None):
        if book_name:
            booking_file_path = f"../Booking/{book_name}.xlsx"
        else:
            booking_file_path = '../Booking/Booking_history.xlsx'
        booking_file_sheet_name = 'histo_order'
        try:
            df = pd.read_excel(booking_file_path, sheet_name=booking_file_sheet_name)
        except FileNotFoundError:
            df = pd.DataFrame(columns=['position', 'type', 'quantité', 'maturité', 'asset', 'price asset', 's-p', 'MtM','strike', 'volatility', 'date heure', 'delta', 'gamma', 'vega', 'theta'])
        position = 'long' if self.quantity>0 else 'short'
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        type = 'asset'
        booking = {'position': position, 'type':type, 'quantité':self.quantity, 'maturité':None, 'asset':self.name, 'price asset':self.St, 's-p': -self.St*lot_size*self.quantity, 'MtM': self.St*lot_size*self.quantity,'strike': None, 'volatility':None, 'date heure':date, 'delta':self.Delta_DF(), 'gamma':None, 'vega':None, 'theta':None}
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

if __name__ == '__main__':
    stock1 = asset_BS(2.240, 0, mu=0.1, sigma=0.2)
    stock1.simu_asset(1/365)
    stock1.plot()
