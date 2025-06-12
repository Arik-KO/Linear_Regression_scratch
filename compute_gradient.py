import numpy as np


def compute_gradient(x, y, w, b):
    m = x.shape[0]

    f = np.dot(w, x) + b
    dw = np.sum((f - y) * x)
    db = np.sum(f - y)
    dw /= m
    db /= m

    return dw, db
