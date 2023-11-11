import statistics
import numpy as np
from Maths import normal_repartition, normal_distribution


def payoff_call_eu(ST, K):
    return(max(ST-K, 0))
def payoff_put_eu(ST, K):
    return(max(K-ST, 0))
def payoff_call_asian(St, K):
    Avg_path = statistics.mean(St)
    return(max(Avg_path - K, 0))
def payoff_put_asian(St, K):
    Avg_path = statistics.mean(St)
    return(max(K - Avg_path, 0))
def d__(St, K, t, T, r, sigma):
    d_1 = (np.log(St / K) + (r + 0.5 * sigma ** 2) * (T-t)) / (sigma * np.sqrt(T-t))
    d_2 = d_1 - sigma * np.sqrt(T-t)
    return d_1, d_2
def close_formulae_call_eu(St, K, t, T, r, sigma):
    d_1, d_2 = d__(St, K, t, T, r, sigma)
    return St * normal_repartition(d_1) - K * np.exp(-r *(T-t)) * normal_repartition(d_2)
def close_formulae_put_eu(St, K, t, T, r, sigma):
    d_1, d_2 = d__(St, K, t, T, r, sigma)
    return -St * normal_repartition(-d_1) + K * np.exp(-r *(T-t)) * normal_repartition(-d_2)
def delta_option_eu(type, St, K, t, T, r, sigma):
    d_1, d_2 = d__(St, K, t, T, r, sigma)
    if type == 'Call EU':
        return normal_repartition(d_1)
    elif type == 'Put EU':
        return -normal_repartition(-d_1)
def gamma_option_eu(type, St, K, t, T, r, sigma):
    d_1, d_2 = d__(St, K, t, T, r, sigma)
    derive_d_1 = 1 / (St*sigma * np.sqrt(T - t))
    if type == 'Call EU':
        return normal_distribution(d_1) / (St*sigma * np.sqrt(T - t))
    elif type == 'Put EU':
        return derive_d_1*normal_distribution(-d_1)