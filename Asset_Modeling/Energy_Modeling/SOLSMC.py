import matplotlib.pyplot as plt
import numpy as np

# Swing option parameters
S0 = 100   # Initial price
K = 100    # Strike price
T = 10.0 / 365  # Time to maturity (in years)
r = 0.05   # Risk-free interest rate
sigma = 0.2  # Volatility
N = 10     # Number of time steps
M = 10000  # Number of Monte Carlo paths
exercises = 3  # Maximum number of exercises allowed

# Time discretization
dt = T / N
discount_factor = np.exp(-r /365)

# Simulate price paths
np.random.seed(42)
price_paths = np.zeros((M, N + 1))
price_paths[:, 0] = S0
for t in range(1, N + 1):
    z = np.random.standard_normal(M)
    price_paths[:, t] = price_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

# Payoff function
def payoff(S, K):
    return np.maximum(S - K, 0)

# LSMC pricing function
def price_swing_option_lsmc(price_paths, K, exercises):
    M, N = price_paths.shape
    option_values = np.zeros(M)
    remaining_exercises = np.full(M, exercises)

    # Initialize option values at maturity (at last time step)
    option_values[:] = payoff(price_paths[:, -1], K)

    for t in reversed(range(1, N)):
        # Only consider paths that have remaining exercises and are in the money
        in_the_money = payoff(price_paths[:, t], K) > 0
        valid_indices = np.where(in_the_money & (remaining_exercises > 0))[0]

        if len(valid_indices) > 0:
            # Values of option and corresponding prices for valid paths
            Y = option_values[valid_indices] * discount_factor
            X = price_paths[valid_indices, t]

            # Fit polynomial to the continuation value
            poly = np.polyfit(X, Y, deg=2)
            continuation_value = np.polyval(poly, X)

            # Calculate the immediate exercise value for the valid paths
            exercise_value = payoff(X, K)

            # Decide whether to exercise or continue based on continuation value
            exercise_decision = exercise_value > continuation_value

            # Update option values based on the exercise decision
            option_values[valid_indices[exercise_decision]] = exercise_value[exercise_decision]
            remaining_exercises[valid_indices[exercise_decision]] -= 1  # Reduce the count of remaining exercises

    # Return the average of the option values
    return np.mean(option_values)

# Price the swing option using LSMC
for i in range(M):
    plt.plot(price_paths[i], label=f'Path {i+1}')
plt.title('Asset price paths')
plt.show()
swing_option_price_lsmc = price_swing_option_lsmc(price_paths, K, exercises)
print(f"Swing Option Price (LSMC): {swing_option_price_lsmc:.2f}")
