from BM_def import BM
import numpy as np
from Graphics import plot_2d
import matplotlib.pyplot as plt
from Actif_stoch_BS import simu_actif

if __name__ == '__main__':
    N = 100
    T = 1
    Nmc = 1000
    time = np.linspace(0, T, N+1)
    title = f'SImulation de {Nmc} prix d actif'
    x_axis = 'Temps'
    y_axis = 'Valeur'
    for i in range(Nmc):
        W_t = BM(N, T)
        St = simu_actif(10, N, 1, 0.1, 0.3)
        plot_2d(time, St, title, x_axis, y_axis, False)
    plt.show()

