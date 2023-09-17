import matplotlib.pyplot as plt

def plot_2d(x_, y_, titre, x_axis, y_axis, isShow=True):
    plt.plot(x_, y_)
    plt.title(titre)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if isShow:
        plt.show()