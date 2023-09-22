def payoff_call_eu(ST, K):
    return(max(ST-K, 0))
def payoff_put_eu(ST, K):
    return(max(K-ST, 0))