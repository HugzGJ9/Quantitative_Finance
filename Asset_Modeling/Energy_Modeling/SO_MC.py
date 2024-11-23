import numpy as np

# Swing option parameters
S0 = 100  # Initial price of the underlying asset
K = 100  # Strike price (fixed for each exercise)
T = 10.0/365  # Time to maturity (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
N = 10  # Number of time steps
M = 10  # Number of Monte Carlo paths
exercises = 3  # Maximum number of exercises allowed

# Time discretization
dt = T / N
discount_factor = np.exp(-r * dt)

# Simulate geometric Brownian motion paths
np.random.seed(42)
price_paths = np.zeros((M, N + 1))
price_paths[:, 0] = S0
for t in range(1, N + 1):
    z = np.random.standard_normal(M)
    price_paths[:, t] = price_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)


# Function to calculate the payoff for a single exercise
def payoff(S, K):
    return np.maximum(S - K, 0)

# Function to calculate the swing option value
def price_swing_option(price_paths, K, exercises):
    M, N = price_paths.shape
    option_values = np.zeros(M)

    # For each path, we need to track remaining exercises
    for i in range(M):
        remaining_exercises = exercises
        path_payoff = 0

        # Go through each time step and decide whether to exercise
        for t in range(1, N):
            if remaining_exercises > 0:
                exercise_value = payoff(price_paths[i, t], K)

                # If it's beneficial to exercise, do so
                if exercise_value > 0:
                    path_payoff += exercise_value * (discount_factor ** t)
                    remaining_exercises -= 1

        option_values[i] = path_payoff

    # Return the average payoff across all Monte Carlo paths
    return np.mean(option_values)


# Price the swing option
swing_option_price = price_swing_option(price_paths, K, exercises)
print(f"Swing Option Price: {swing_option_price:.2f}")
