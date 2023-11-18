import matplotlib.pyplot as plt

import payoffs


def plot_2d(x_, y_, titre, x_axis, y_axis, isShow=True):
    plt.plot(x_, y_)
    plt.title(titre)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if isShow:
        plt.show()

def display_payoff(Option_type, Strike, Strike2=None):

    if Option_type == "Call EU":
        payoff_function = payoffs.payoff_call_eu
    elif Option_type == "Put EU":
        payoff_function = payoffs.payoff_put_eu
    elif Option_type == "Call Spread":
        payoff_function = payoffs.payoff_call_spread

    if Strike2:
        start = Strike*0.5 if Strike<Strike2 else Strike2*0.5
        end = Strike*1.5 if Strike>Strike2 else Strike2*1.5
    else:
        start = Strike * 0.5
        end = Strike * 1.5
    ST = list(range(round(start), round(end)))
    payoff = []
    if Strike2:
        for i in ST:
            payoff.append(payoff_function(i, Strike, Strike2))
    else:
        for i in ST:
            payoff.append(payoff_function(i, Strike))
    plot_2d(ST, payoff, f"{Option_type} payoff", "Asset price", "Payoff", isShow=True)