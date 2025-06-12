import copy
import math


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    j_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dw, db = gradient_function(x, y, w, b)
        w = w - alpha*dw
        b = b - alpha*db

        if i < 10000:
            cost = cost_function(x, y, w, b)
            j_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return w, b, j_history, w_history
