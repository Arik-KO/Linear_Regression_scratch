import numpy as np


def compute_cost(x, y, w, b):
    m = x.shape[0]
    f = np.dot(w, x) + b
    total_cost = np.sum((f - y) ** 2)
    total_cost /= (2*m)
    return total_cost
