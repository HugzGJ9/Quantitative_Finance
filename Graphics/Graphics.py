import matplotlib.pyplot as plt
from Options import payoffs


def plot_2d(x_, y_, x_axis, y_axis, isShow=True, title=None, legend=None):
    plt.plot(x_, y_, label='Delta vs Range', color='blue', linestyle='-', linewidth=2)

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    if legend:
        plt.legend(legend)
    if isShow:
        plt.grid(True)
        plt.show()

def display_payoff_eu(Option_type, Strike:(list, int), plot=True):

    if Option_type == "Call EU":
        payoff_function = payoffs.payoff_call_eu
    elif Option_type == "Put EU":
        payoff_function = payoffs.payoff_put_eu

    if type(Strike) == list:
        start = min(Strike)*0.5
        end = max(Strike)*1.5
    else:
        start = Strike * 0.5
        end = Strike * 1.5
    ST = list(range(round(start), round(end)))
    payoff = []

    for i in ST:
        payoff.append(payoff_function(i, Strike))
    plot_2d(ST, payoff, "Asset price", "Payoff", isShow=plot, title=f"{Option_type} payoff")

    return [ST, payoff]

def correl_plot(df, x_label, y_label, title):
    df.plot()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
    return