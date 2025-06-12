import numpy as np


def compute_cost(x, y, w, b):
    m, _ = x.shape
    A = np.dot(x, w.T) + b
    cost = np.sum((A - y) ** 2)
    total_cost = cost / (2 * m)
    return total_cost
