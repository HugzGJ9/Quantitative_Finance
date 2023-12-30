from Actif_stoch_BS import simu_actif
from Graphics import plot_2d


class asset_BS():
    def __init__(self, S0, quantity):
        self.S0 = S0
        self.St = S0
        self.quantity = quantity
        self.history = [S0]
        self.mu = 0.1
        self.sigma = 0.2
        self.t = 0
    def simu_asset(self, T)->None:
        St = simu_actif(self.St, T, self.t, T, self.mu, self.sigma)
        St.pop(0)
        for st in St:
            self.history.append(st)
        self.St = self.history[-1]
    def pnl(self)->float:
        return self.St - self.S0
    def plot(self):
        plot_2d(list(range(len(self.history))), self.history, x_axis='t', y_axis='asset price', isShow=True)
    def Delta_DF(self):
        return 1*self.quantity
    def Gamma_DF(self):
        return 0
    def Vega_DF(self):
        return 0
    def Theta_DF(self):
        return 0