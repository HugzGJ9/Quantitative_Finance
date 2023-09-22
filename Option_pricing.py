from BM_def import BM
import numpy as np
from Graphics import plot_2d
import matplotlib.pyplot as plt
from Actif_stoch_BS import simu_actif
from Actif_stoch_BS import option_eu_mc

class Option_eu:
    def __init__(self, type, St, K, t, T, sigma):
        self.type = type
        self.St = St
        self.K = K
        self.t = t
        self.T = T
        self.sigma = sigma

    def option_eu_mc(self):
        prix_option = 0
        Nmc = 10000
        print(self.St)
        for i in range(Nmc):
            prix_actif = simu_actif(self.St, 100, self.T, 0.1, self.sigma)
            prix_option += max(prix_actif[-1] - self.K, 0)
        prix_option = prix_option / Nmc
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


    call_eu1 = Option_eu('call eu', St, K, t, T, 0.3)
    print(call_eu1.option_eu_mc())
    # print(option_eu_mc(St, T, K, Nmc))

    # for i in range(Nmc):
    #     St = simu_actif(10, N, T, 0.1, 0.3)
    #     plot_2d(time, St, title, x_axis, y_axis, False)
    # plt.show()

