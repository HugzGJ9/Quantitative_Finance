import numpy as np

S0 = 100
K = 100
T = 10.0/365
r = 0.05
sigma = 0.2
N = 10
exercises = 3
dt = T / N
discount_factor = np.exp(-r * dt)

u = np.exp(sigma * np.sqrt(dt))
d = 1 / u
q = (np.exp(r * dt) - d) / (u - d)

def build_binomial_tree(S0, u, d, N):
    tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
    return tree

price_tree = build_binomial_tree(S0, u, d, N)

def payoff(S, K):
    return np.maximum(S - K, 0)

def price_swing_option_binomial(price_tree, K, exercises):
    # Initialize the option values at the final nodes
    option_values = np.zeros((N + 1, N + 1, exercises + 1))

    # Loop backward through the tree
    for t in reversed(range(N)):
        for i in range(t + 1):
            for e in range(1, exercises + 1):
                hold_value = discount_factor * (
                            q * option_values[i, t + 1, e] + (1 - q) * option_values[i + 1, t + 1, e])
                exercise_value = payoff(price_tree[i, t], K) + discount_factor * (
                            q * option_values[i, t + 1, e - 1] + (1 - q) * option_values[i + 1, t + 1, e - 1])
                option_values[i, t, e] = np.maximum(hold_value, exercise_value)

    return option_values[0, 0, exercises]


swing_option_price_binomial = price_swing_option_binomial(price_tree, K, exercises)
print(f"Swing Option Price (Binomial): {swing_option_price_binomial:.2f}")
