import numpy as np


def next_price_idx(P_idx, up=True):
    if up:
        return min(P_idx + 1, M - 1)
    else:
        return max(P_idx - 1, 0)

def gas_storage_price(S_0, P_0):
    S_idx = np.argmin(np.abs(S_levels - S_0))
    P_idx = np.argmin(np.abs(P_values - P_0))

    return V[S_idx, P_idx]


T = 1.0
dt = 0.01
N = int(T / dt)
P_max = 10
P_min = 1
dP = 0.1
sigma = 0.3
mu = 0.05

P_values = np.arange(P_min, P_max + dP, dP)
M = len(P_values)

# Storage levels: 0%, 50%, 100%
S_levels = np.array([0, 0.5, 1])

V = np.zeros((3, M))

# Backward iteration in time to compute value function
for t in reversed(range(N)):
    for i, S in enumerate(S_levels):
        for j, P in enumerate(P_values):
            # Holding value (no injection or withdrawal)
            V_hold = V[i, j]
            if i < 2:
                V_inject = V[i + 1, j]
            else:
                V_inject = -np.inf

            if i > 0:
                V_withdraw = V[i - 1, j] + P
            else:
                V_withdraw = -np.inf

            V[i, j] = max(V_hold, V_inject, V_withdraw)

            # Gas price dynamics (using GBM approximation)
            P_up_idx = next_price_idx(j, up=True)
            P_down_idx = next_price_idx(j, up=False)

            # Update value with time decay and price dynamics (Euler method)
            dV_dt = -mu * P_values[j] * (V[i, P_up_idx] - V[i, P_down_idx]) / (2 * dP)
            dV_dP = 0.5 * sigma ** 2 * P_values[j] ** 2 * (V[i, P_up_idx] - 2 * V[i, j] + V[i, P_down_idx]) / (dP ** 2)

            dV_dt = np.clip(dV_dt, -1000, 1000)
            dV_dP = np.clip(dV_dP, -1000, 1000)

            V[i, j] += dt * (dV_dt + dV_dP)

S_0 = 0.0
P_0 = 5.0

price_of_storage = gas_storage_price(S_0, P_0)
print(f"Price of the gas storage facility: ${price_of_storage:.2f}")
