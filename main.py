import BM_def
import numpy as np
from Graphics import plot_2d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 100
    T = 1
    Nmc = 1000
    time = np.linspace(0, T, N+1)
    title = f'SImulation de {Nmc} mouvement brownien'
    x_axis = 'Temps'
    y_axis = 'Valeur'
    for i in range(Nmc):
        W_t = BM_def.BM(N, T)
        plot_2d(time, W_t, title, x_axis, y_axis, False)
    plt.show()


