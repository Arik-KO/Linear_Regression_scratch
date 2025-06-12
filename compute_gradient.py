import numpy as np


def compute_gradient(x, y, w, b):
    m, _ = x.shape
    A = np.dot(x, w.T) + b
    dz = A - y
    dw = np.dot(x, w.T) + b
    db = np.sum(dz)
    dw /= m
    db /= m

    return db, dw
