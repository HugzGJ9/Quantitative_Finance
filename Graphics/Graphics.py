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

def plot_multiple_lists(*price_paths):
        n = len(price_paths)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, paths, idx in zip(axes, price_paths, range(n)):
            for path in paths:
                ax.plot(path)
            ax.set_title(f'PNL Paths Plot {idx + 1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('PNL')
            ax.grid(True)

        plt.tight_layout()
        plt.show()


def DAauctionplot(df, title='Price Curve and Value Histogram', show=True):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot price curve (line)
    ax1.plot(df['datetime'], df['price'],
             label='Price',
             linestyle='-',
             linewidth=2,
             color='blue')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Datetime')

    # Create second axis for value bars
    ax2 = ax1.twinx()

    # Compute dynamic bar width (optional polish)
    if len(df['datetime']) > 1:
        delta = (df['datetime'].iloc[1] - df['datetime'].iloc[0]).total_seconds()
        bar_width = delta / (24 * 60 * 60) * 0.8  # fraction of a day
    else:
        bar_width = 0.03

    # Plot histogram as bar
    ax2.bar(df['datetime'], df['value'],
            alpha=0.4,
            color='orange',
            label='Value',
            width=bar_width)
    ax2.set_ylabel('Value', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Grid and layout
    ax1.grid(True, which='major', linestyle='--', alpha=0.6)
    fig.autofmt_xdate()
    plt.title(title)
    fig.tight_layout()

    # Optional legend (blue + orange)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    if lines_1 or lines_2:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    if show:
        plt.show()
    return fig

if __name__ == '__main__':
    data = pd.read_excel('h:/Downloads/DApriceES2023.xlsx')

