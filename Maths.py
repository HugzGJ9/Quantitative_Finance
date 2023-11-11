import numpy as np
import math
from scipy.stats import norm
def norm_():
    rand = np.random.randn()
    return rand
def normal_repartition(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))
def normal_distribution(x):
    return norm.pdf(x)
