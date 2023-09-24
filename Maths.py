import numpy as np
import math
def norm():
    rand = np.random.randn()
    return rand
def normal_repartition(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))