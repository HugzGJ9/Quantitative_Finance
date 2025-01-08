import matplotlib.pyplot as plt
from matplotlib import pyplot

from Options import payoffs
import pandas as pd
def plot_power_correl_vsCountries(dataframe, country, loop, plot=True):
    dataframe.plot(linestyle='-', linewidth=2, title=f'{country}vs Neighbour countries {loop}')
    if plot:
        plt.grid(True)
        plt.legend()
        plt.show()
    return
def plot_2d(x_, y_, x_axis=None, y_axis=None, plot=True, title=None, legend=None):
    if plot:
        plt.plot(x_, y_, label=title, linestyle='-', linewidth=2, color = 'blue',)
    else:
        plt.plot(x_, y_, label=title, linestyle='-', linewidth=2)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if legend:
        plt.legend(legend)
    if plot:
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
    plot_2d(ST, payoff, "Asset price", "Payoff", plot=plot, title=f"{Option_type} payoff")

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

def plotGreek(St, Greek_Option, list_delta, range_st, greek):
    plt.plot(range_st, list_delta, label=f'{greek} vs Range', color='blue', linestyle='-', linewidth=2)
    plt.plot(St, Greek_Option, 'x', label=f'Option {greek} at Specific Points',
             color='red', markersize=10, markeredgewidth=2)
    plt.title(f'{greek} of the Option vs. Underlying Asset Price')
    plt.xlabel('Underlying Asset Price (St)')
    plt.ylabel(f'Option {greek}')
    plt.legend()
    plt.grid(True)
    plt.show()
def plotPnl(list_pnl, range_st, n_order_pnl):
    plt.plot(range_st, list_pnl, label=f'{n_order_pnl} vs Range', color='blue', linestyle='-', linewidth=2)
    plt.plot(range_st[int(len(range_st)/2)], 0, 'x', label=f'Spot',
             color='red', markersize=10, markeredgewidth=2)
    plt.title(f'{n_order_pnl} of the Option vs. Underlying Asset Price')
    plt.xlabel('Underlying Asset Price (St)')
    plt.ylabel(f'Option {n_order_pnl}')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    data = pd.read_excel('h:/Downloads/DApriceES2023.xlsx')

