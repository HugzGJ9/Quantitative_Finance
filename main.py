import BM_def
import matplotlib.pyplot as plt

if __name__ == '__main__':
    time, W_t = BM_def.BM(100, 1)
    plt.plot(time, W_t)
    plt.title("Simulation du mouvement brownien")
    plt.show()
