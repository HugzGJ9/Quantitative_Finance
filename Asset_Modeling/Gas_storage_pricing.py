import numpy as np


# Helper function for the GBM gas price dynamics
def next_price_idx(P_idx, up=True):
    if up:
        return min(P_idx + 1, M - 1)
    else:
        return max(P_idx - 1, 0)

# Function to compute the price of the gas storage facility
def gas_storage_price(S_0, P_0):
    # Find the closest indices for the initial storage level and price
    S_idx = np.argmin(np.abs(S_levels - S_0))
    P_idx = np.argmin(np.abs(P_values - P_0))

    # Return the corresponding value from the value function at time t=0
    return V[S_idx, P_idx]


# Parameters
T = 1.0  # Time horizon in years
dt = 0.01  # Time step size
N = int(T / dt)  # Number of time steps
P_max = 10  # Maximum gas price
P_min = 1  # Minimum gas price
dP = 0.1  # Gas price step size
sigma = 0.3  # Volatility of gas prices
mu = 0.05  # Drift of gas prices

# Discretize gas prices
P_values = np.arange(P_min, P_max + dP, dP)
M = len(P_values)

# Storage levels: 0%, 50%, 100%
S_levels = np.array([0, 0.5, 1])

# Value function initialization (final values are 0 as there's no terminal value)
V = np.zeros((3, M))

# Backward iteration in time to compute value function
for t in reversed(range(N)):
    for i, S in enumerate(S_levels):
        for j, P in enumerate(P_values):
            # Holding value (no injection or withdrawal)
            V_hold = V[i, j]

            # Inject gas if possible
            if i < 2:
                V_inject = V[i + 1, j]
            else:
                V_inject = -np.inf  # No injection if full

            # Withdraw gas if possible
            if i > 0:
                V_withdraw = V[i - 1, j] + P  # Value from selling gas at current price
            else:
                V_withdraw = -np.inf  # No withdrawal if empty

            # Control decision: inject, withdraw, or hold
            V[i, j] = max(V_hold, V_inject, V_withdraw)

            # Gas price dynamics (using GBM approximation)
            P_up_idx = next_price_idx(j, up=True)
            P_down_idx = next_price_idx(j, up=False)

            # Update value with time decay and price dynamics (Euler method)
            dV_dt = -mu * P_values[j] * (V[i, P_up_idx] - V[i, P_down_idx]) / (2 * dP)
            dV_dP = 0.5 * sigma ** 2 * P_values[j] ** 2 * (V[i, P_up_idx] - 2 * V[i, j] + V[i, P_down_idx]) / (dP ** 2)

            # Cap value updates to prevent instability
            dV_dt = np.clip(dV_dt, -1000, 1000)
            dV_dP = np.clip(dV_dP, -1000, 1000)

            # Ensure the final value doesn't explode
            V[i, j] += dt * (dV_dt + dV_dP)

# Example: Compute the price of the gas storage facility for S_0 = 50% (half-full) and P_0 = $5
S_0 = 0.0  # Initial storage level (50% full)
P_0 = 5.0  # Initial gas price

price_of_storage = gas_storage_price(S_0, P_0)
print(f"Price of the gas storage facility: ${price_of_storage:.2f}")
